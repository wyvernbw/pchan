#![feature(try_blocks)]
#![feature(iter_array_chunks)]
#![feature(clone_from_ref)]

pub(crate) mod asm_dump_widget;
#[path = "./emu-task.rs"]
pub(crate) mod emu_task;
#[path = "./lipgloss-colors.rs"]
pub(crate) mod lipgloss_colors;
pub(crate) mod mem_widget;
pub(crate) mod tty_widget;

use std::borrow::Cow;
use std::process::Stdio;
use std::sync::Arc;

use color_eyre::Result;
use flume::{Receiver, Sender};
use manatui::prelude::{strum::EnumCount, *};
use manatui::ratatui::crossterm::event::Event;
use manatui::ratatui::text::{Line, Span};
use manatui::tea;
use manatui::tea::Effect;
use manatui::tea::focus::{EventOutcome, FocusGroup};
use manatui::tea::observe::AreaRef;
use manatui::utils::keyv2;
use manatui_tea_ui::components::list::{List, ListViewCompact};
use pchan_emu::cpu::{Cpu, reg_str};
use pchan_emu::dynarec_v2::PipelineV2Stage;
use pchan_utils::{hex, setup_tracing_file_only};
use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;

use crate::asm_dump_widget::{AsmDump, AsmDumpView};
use crate::emu_task::EmuTaskState;
use crate::emu_task::{EmuRequest, EmuResponse, EmuTask, EmuTaskHandle};
use crate::mem_widget::{MemView, MemViewState};
use crate::tty_widget::{TtyView, TtyViewState};

pub(crate) type Chan<T> = (Sender<T>, Receiver<T>);

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    setup_tracing_file_only();
    tea::run()
        .init(Model::init)
        .view(view)
        .update(Model::update)
        .event_msg(Msg::Event)
        .quit_signal(|_, msg| matches!(msg, Msg::Quit))
        .enable_mouse(true)
        .run()
        .await?;

    Ok(())
}

struct Model {
    dbg_page:        DbgPage,
    emu_handle:      EmuTaskHandle,
    emu_join_handle: JoinHandle<Result<()>>,
    emu_stage:       PipelineV2Stage,
    emu_task_state:  EmuTaskState,
    emu_cpu:         Box<Cpu>,
    tty:             Vec<Arc<str>>,
    objdump:         Option<Arc<str>>,
}

#[derive(Debug, Clone, Copy)]
enum DbgPageFocus {
    Ops,
    CpuViewer,
    MemViewer,
    TtyViewer,
    AsmViewer,
}

struct DbgPage {
    decoded_ops:            Option<Arc<[Line<'static>]>>,
    focus:                  FocusGroup<DbgPageFocus>,
    decoded_ops_list:       List,
    cpu_gpr_list:           List,
    cpu_viewer_show_zeroes: bool,
    cpu_viewer_show_cops:   bool,
    cpu_viewer_gpr_rect:    AreaRef,
    mem_view:               MemViewState,
    tty_view:               TtyViewState,
    asm_dump:               AsmDump,
}

#[derive(Debug, Clone)]
enum Msg {
    Quit,
    Event(Event),
    EmuRes(EmuResponse),
    TtyLine(Arc<str>),
}

impl tea::Message for Msg {
    type Model = Model;
}

impl tea::Model for Model {}

impl Model {
    async fn init() -> (Self, Effect<Msg>) {
        let (emu_handle, emu_join_handle) = match EmuTask::spawn() {
            Ok(handle) => handle,
            Err(err) => panic!("{err:?}"),
        };
        let rx = emu_handle.res_chan.1.clone();
        let tty_rx = emu_handle.tty_rx.clone();

        (
            Model {
                dbg_page: DbgPage {
                    decoded_ops:            None,
                    decoded_ops_list:       List::new().set_tab_navigation(false),
                    cpu_gpr_list:           List::new().set_tab_navigation(false),
                    focus:                  FocusGroup::new().set_wrap_around(true),
                    cpu_viewer_show_zeroes: true,
                    cpu_viewer_show_cops:   false,
                    cpu_viewer_gpr_rect:    AreaRef::empty(),
                    mem_view:               MemViewState::new(),
                    tty_view:               TtyViewState::new(),
                    asm_dump:               AsmDump::new(),
                },
                emu_handle,
                emu_join_handle,
                emu_stage: PipelineV2Stage::Uninit,
                emu_task_state: EmuTaskState::Paused,
                emu_cpu: Box::default(),
                tty: vec![],
                objdump: None,
            },
            Effect::new(async move |tx| {
                loop {
                    tokio::select! {
                        Ok(res) = rx.recv_async() => {
                            _ = tx.send_async(Msg::EmuRes(res)).await;
                        }
                        Ok(log) = tty_rx.recv_async() => {
                            _ = tx.send_async(Msg::TtyLine(log)).await;
                        }
                    }
                }
            }),
        )
    }

    fn set_emu_stage(self, stage: PipelineV2Stage) -> Self {
        Self {
            emu_stage: stage,
            ..self
        }
    }

    fn set_emu_task_state(self, state: EmuTaskState) -> Self {
        Self {
            emu_task_state: state,
            ..self
        }
    }

    fn no_effect(self) -> (Self, Effect<Msg>) {
        (self, Effect::none())
    }

    async fn update(mut self, msg: Msg) -> (Self, Effect<Msg>) {
        match msg {
            Msg::Quit => unreachable!(),
            Msg::Event(event) => self.handle_event(event).await,
            Msg::EmuRes(res) => match res {
                EmuResponse::StageUpdate(stage) => self.set_emu_stage(stage).no_effect(),
                EmuResponse::StateUpdate(state) => self.set_emu_task_state(state).no_effect(),
                EmuResponse::Compiled(pc, decoded_ops, func) => {
                    let decoded_ops = Some(
                        decoded_ops
                            .iter()
                            .enumerate()
                            .map(|(idx, op)| {
                                let line = Line::from_iter([
                                    Span::raw(hex(idx as u32 * 4 + pc).to_string())
                                        .style(Style::new().dim()),
                                    Span::raw(" "),
                                    Span::raw(op.to_string()),
                                ]);
                                let style = match idx.is_multiple_of(2) {
                                    true => Style::new(),
                                    false => Style::new().fg(Color::from_u32(0xeeeeee)),
                                };

                                line.style(style)
                            })
                            .collect(),
                    );
                    self.dbg_page = DbgPage {
                        decoded_ops,
                        ..self.dbg_page
                    };
                    (
                        self,
                        Effect::new(async move |tx| {
                            let _ = try {
                                let mut file =
                                    tokio::fs::File::create("/tmp/pchan-dump.bin").await?;
                                file.write_all(func.buffer().as_ref()).await?;
                                let mut cmd = tokio::process::Command::new("objdump");
                                cmd.args([
                                    "-D",
                                    "-b",
                                    "binary",
                                    "-m",
                                    "aarch64",
                                    "/tmp/pchan-dump.bin",
                                ])
                                .stdout(Stdio::piped());
                                cmd.spawn()?;
                                let output = cmd.output().await?;
                                let output = String::from_utf8_lossy(&output.stdout);
                                let output = Arc::<str>::from(output);
                                _ = tx.send(Msg::EmuRes(EmuResponse::ObjDump(output)));
                            };
                        }),
                    )
                }
                EmuResponse::CpuUpdate(cpu) => {
                    self.emu_cpu = cpu;
                    self.no_effect()
                }
                EmuResponse::ObjDump(asm) => {
                    self.objdump = Some(asm);
                    self.no_effect()
                }
            },
            Msg::TtyLine(line) => {
                self.tty.push(line);
                self.no_effect()
            }
        }
    }

    async fn handle_event(mut self, event: Event) -> (Self, Effect<Msg>) {
        match event {
            keyv2!(ctrl + 'c') => {
                self.send_request(EmuRequest::Quit).await;
                (self, Effect::msg(Msg::Quit))
            }
            keyv2!('n') => {
                self.send_request(EmuRequest::Step).await;
                self.no_effect()
            }
            event => {
                match self.dbg_page.focus.update(&event) {
                    EventOutcome::Consumed(focus) => {
                        self.dbg_page.focus = focus;
                        self.build_focus();
                        return self.no_effect();
                    }
                    EventOutcome::Unhandled(focus) => {
                        self.dbg_page.focus = focus;
                    }
                }

                self.dbg_page.tty_view = self.dbg_page.tty_view.update(&event);
                (self.dbg_page.decoded_ops_list, self.dbg_page.focus) = self
                    .dbg_page
                    .focus
                    .pipe(self.dbg_page.decoded_ops_list.update(&event));
                (self.dbg_page.cpu_gpr_list, self.dbg_page.focus) = self
                    .dbg_page
                    .focus
                    .pipe(self.dbg_page.cpu_gpr_list.update(&event));
                self.dbg_page.mem_view = self.dbg_page.mem_view.update(&event);
                self.dbg_page.asm_dump = self.dbg_page.asm_dump.update(&event);

                self.build_focus();

                match (self.dbg_page.focus.tag(), event) {
                    (Some(DbgPageFocus::CpuViewer), keyv2!('s')) => {
                        self.dbg_page.cpu_viewer_show_zeroes =
                            !self.dbg_page.cpu_viewer_show_zeroes;
                    }
                    (Some(DbgPageFocus::CpuViewer), keyv2!('c')) => {
                        self.dbg_page.cpu_viewer_show_cops = !self.dbg_page.cpu_viewer_show_cops;
                    }
                    _ => {}
                }

                self.no_effect()
            }
            _ => (self, Effect::none()),
        }
    }

    fn build_focus(&mut self) {
        self.dbg_page
            .focus
            .items()
            .next((&self.dbg_page.decoded_ops_list, DbgPageFocus::Ops))
            .next((&self.dbg_page.cpu_gpr_list, DbgPageFocus::CpuViewer))
            .next((&self.dbg_page.mem_view, DbgPageFocus::MemViewer))
            .next((&self.dbg_page.tty_view, DbgPageFocus::TtyViewer))
            .next((&self.dbg_page.asm_dump, DbgPageFocus::AsmViewer))
            .commit();
    }

    async fn send_request(&self, req: EmuRequest) {
        _ = self.emu_handle.req_chan.0.send_async(req).await;
    }
}

async fn view(model: &Model) -> View {
    ui! {
        <Block .rounded .title="+ 🐷🎗️ P-ちゃん dbg +" Width::grow() Height::grow() Direction::Horizontal>
            <Block Width::percentage(25) Height::grow() MaxWidth::percentage(25)>
                <Instructions .model={model} Height::grow()/>
                <Summary .model={model} Width::grow()/>
            </Block>
            <Block Height::grow()>
                <CpuViewer .model={model} />
            </Block>
            <Block Height::grow() Width::grow() MaxWidth::percentage(50)>
                <MemView .view={&model.emu_handle.dbg_view} .state={&model.dbg_page.mem_view}/>
            </Block>
            <Block Height::grow() Width::grow()>
                <TtyView .state={&model.dbg_page.tty_view} .tty={&model.tty} />
                <AsmDumpView
                    .state={&model.dbg_page.asm_dump}
                    .asm={model.objdump.clone().unwrap_or_default()}
                    Width::grow() Height::grow()
                />
            </Block>
            <CopView .model={model}/>
        </Block>
    }
}

#[subview]
fn summary(model: &Model) -> View {
    let stage = model.emu_stage;
    let stage_idx = (model.emu_stage as u8).saturating_sub(1);
    let max_stage_idx = PipelineV2Stage::COUNT as u8 - 2;
    let ratio = stage_idx as f64 / max_stage_idx as f64;
    let title = format!("{:?} n - step / r - run", model.emu_task_state);
    ui! {
        <Block .rounded .title_bottom={Line::raw(title)} {Padding::new(2, 2, 1, 1)}>
            <Block
                Direction::Horizontal Gap(1) MainJustify::SpaceBetween Width::fixed(24)
            >
                <LineGauge
                    .ratio={ratio}
                    .filled_style={Style::new().fg(Color::Green)}
                    .unfilled_style={Style::new().dim()}
                    Height::fixed(1) Width::grow()
                />
                <Text>
                    "{stage}"
                </Text>
            </Block>
        </Block>
    }
}

#[subview]
fn instructions(model: &Model) -> View {
    let focused = matches!(model.dbg_page.focus.tag(), Some(DbgPageFocus::Ops));
    let border_style = border_style_focus(focused);

    let Some(instructions) = &model.dbg_page.decoded_ops else {
        return ui! {
            <Block Width::grow()>
                <Text>"Opcodes"</Text>
                <Block .rounded .border_style={border_style} Height::grow() Width::grow()>
                    <Block .style={Style::new().dim()} Center Width::grow() Height::grow()>
                        " 何も "
                    </Block>
                </Block>
            </Block>
        };
    };

    ui! {
        <Block Width::grow()>
            <Text>"Opcodes"</Text>
            <Block .rounded .border_style={border_style} .title="+ mips dump +" Height::grow() Width::grow()>
                <ListViewCompact
                    .state={&model.dbg_page.decoded_ops_list}
                    .items={instructions.iter().cloned()}
                    .highlight_style={Style::new().bg(Color::Green).fg(Color::Black)}
                    Height::grow()
                />
            </Block>
        </Block>
    }
}

#[subview]
fn hseparator(
    style: Option<Style>,
    #[builder(default = BorderType::Plain)] border_type: BorderType,
) -> View {
    ui! {
        <Block
            .style={style.unwrap_or_default()}
            .borders={Borders::TOP}
            .border_type={border_type}
            Width::grow() Height::fixed(1)
        />
    }
}

fn border_style_focus(focused: bool) -> Style {
    match focused {
        true => Style::new().green(),
        false => Style::new().dim(),
    }
}

fn make_register_list<'a>(
    i: impl Iterator<Item = &'a u32> + Clone,
    reg_to_str: impl Fn(usize) -> Cow<'a, str>,
    show_zeroes: bool,
    column_width: usize,
) -> impl Iterator<Item = Line<'static>> {
    i.clone()
        .enumerate()
        .filter(move |(_, value)| show_zeroes || **value != 0)
        .map(move |(reg, value)| {
            let style = match *value == 0 {
                true => Style::new().dim(),
                false => Style::default(),
            };
            let style = match reg.is_multiple_of(2) {
                true => style,
                false => style.fg(Color::from_u32(0xeeeeee)),
            };
            let reg = format!("${}", reg_to_str(reg));
            let spacing = column_width.saturating_sub(reg.len());
            Line::from_iter([
                Span::raw(" ".repeat(spacing)),
                Span::raw(reg),
                Span::raw(" "),
                Span::raw(hex(*value).to_string()),
            ])
            .style(style)
        })
        .chain([if show_zeroes {
            Line::default()
        } else {
            let hidden = i.filter(|value| **value == 0).count();
            Line::raw(format!("({hidden} values hidden)")).style(Style::new().dim())
        }])
}

trait BoolOnOff {
    fn onoff(self) -> &'static str;
}

impl BoolOnOff for bool {
    fn onoff(self) -> &'static str {
        match self {
            true => "on",
            false => "off",
        }
    }
}

#[subview]
fn cpu_viewer(model: &Model) -> View {
    let focused = matches!(model.dbg_page.focus.tag(), Some(DbgPageFocus::CpuViewer));
    let border_style = border_style_focus(focused);

    let show_zeroes = model.dbg_page.cpu_viewer_show_zeroes;
    let hi = (model.emu_cpu.hilo >> 32) as u32;
    let lo = model.emu_cpu.hilo as u32;

    let gpr = make_register_list(
        model.emu_cpu.gpr.iter().chain([&hi, &lo]),
        |reg| reg_str(reg as u8).into(),
        show_zeroes,
        7,
    );

    let pc = hex(model.emu_cpu.pc);

    let cop_tooltip = Line::from_iter([
        Span::raw("c ").style(Style::new().dim()),
        Span::raw("toggle cop view").style(Style::new().fg(Color::from_u32(0xeeeeee)).dim()),
    ]);
    let gpr_tooltip = Line::from_iter([
        Span::raw(" s ").style(Style::new().dim()),
        Span::raw("toggle zeroes ").style(Style::new().fg(Color::from_u32(0xeeeeee)).dim()),
    ]);
    let bev = model.emu_cpu.cop0.bev().onoff();
    let isc = model.emu_cpu.cop0.isc().onoff();
    ui! {
        <Block Height::grow()>
            <Block Width::grow()>
                "CPU View"
            </Block>
            <Block Direction::Horizontal Height::grow()>
                <Block
                    .border_style={border_style}
                    .rounded .title="-+ gpr +"
                    .title_bottom={gpr_tooltip}
                    {model.dbg_page.cpu_viewer_gpr_rect.clone()}
                    Padding::new(2, 2, 1, 1) Height::grow()
                >
                    <Text .style={Style::new().bold()}>"    $pc {pc}"</Text>
                    <ListViewCompact
                        .items={gpr}
                        .state={&model.dbg_page.cpu_gpr_list}
                        .highlight_style={Style::new().black().on_green().not_dim()}
                        Height::grow()
                        // MaxHeight::percentage(75)
                    />
                    <Hseparator .border_type={BorderType::LightDoubleDashed} .style={Style::new().dim()} />
                    <Block MaxHeight::percentage(25)>
                        <Text>"bev: {bev}"</Text>
                        <Text>"isc: {isc}"</Text>
                    </Block>
                    <Block>
                        {cop_tooltip.into_view()}
                    </Block>
                </Block>
            </Block>
        </Block>
    }
}

#[subview]
fn cop_view(model: &Model) -> View {
    let focused = matches!(model.dbg_page.focus.tag(), Some(DbgPageFocus::CpuViewer));
    let border_style = border_style_focus(focused);
    let show_zeroes = model.dbg_page.cpu_viewer_show_zeroes;
    let cop0items = make_register_list(
        model.emu_cpu.cop0.reg.iter(),
        |reg| format!("cop0reg{reg}").into(),
        show_zeroes,
        10,
    );
    let cop2items = make_register_list(
        model.emu_cpu.cop2.reg.iter(),
        |reg| format!("cop2reg{reg}").into(),
        show_zeroes,
        10,
    );

    let list = List::new();
    let show_cops = focused && model.dbg_page.cpu_viewer_show_cops;

    let gpr_rect = model.dbg_page.cpu_viewer_gpr_rect.get().unwrap_or_default();
    ui! {
        <Block
            .style={Style::new().on_black()}
            Position::Absolute(Value::Cells(gpr_rect.x + gpr_rect.width), Value::Cells(gpr_rect.y))
        >
        {if show_cops {
            ui! {
                <Block
                    .rounded
                    .title="-+ coprocessors +"
                    .border_style={border_style}
                    .borders={Borders::TOP | Borders::RIGHT | Borders::BOTTOM}
                    Clear
                    Padding::new(1, 3, 1, 1)
                    Direction::Horizontal
                    Gap(2)
                >
                    <ListViewCompact
                        .items={cop0items}
                        .state={&list}
                    />
                    <ListViewCompact
                        .items={cop2items}
                        .state={&list}
                    />
                </Block>
            }
        } else {
            ui! {""}
        }}
        </Block>
    }
}
