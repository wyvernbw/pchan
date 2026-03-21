#![feature(clone_from_ref)]

#[path = "./emu-task.rs"]
pub(crate) mod emu_task;
#[path = "./lipgloss-colors.rs"]
pub(crate) mod lipgloss_colors;

use std::sync::Arc;

use color_eyre::Result;
use flume::{Receiver, Sender};
use manatui::prelude::{strum::EnumCount, *};
use manatui::ratatui::crossterm::event::Event;
use manatui::ratatui::text::{Line, Span};
use manatui::tea;
use manatui::tea::Effect;
use manatui::tea::focus::{EventOutcome, FocusGroup};
use manatui::utils::keyv2;
use manatui_tea_ui::components::list::{List, ListViewCompact};
use pchan_emu::cpu::{Cpu, reg_str};
use pchan_emu::dynarec_v2::PipelineV2Stage;
use pchan_utils::hex;
use tokio::task::JoinHandle;

use crate::emu_task::EmuTaskState;
use crate::{
    emu_task::{EmuRequest, EmuResponse, EmuTask, EmuTaskHandle},
    lipgloss_colors::LIPGLOSS,
};

pub(crate) type Chan<T> = (Sender<T>, Receiver<T>);

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
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
}

struct DbgPage {
    decoded_ops:            Option<Arc<[Line<'static>]>>,
    focus:                  FocusGroup,
    decoded_ops_list:       List,
    cpu_gpr_list:           List,
    cpu_viewer_show_zeroes: bool,
}

#[derive(Debug, Clone)]
enum Msg {
    Quit,
    Event(Event),
    EmuRes(EmuResponse),
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

        (
            Model {
                dbg_page: DbgPage {
                    decoded_ops:            None,
                    decoded_ops_list:       List::new(),
                    cpu_gpr_list:           List::new(),
                    focus:                  FocusGroup::new(),
                    cpu_viewer_show_zeroes: true,
                },
                emu_handle,
                emu_join_handle,
                emu_stage: PipelineV2Stage::Uninit,
                emu_task_state: EmuTaskState::Paused,
                emu_cpu: Box::default(),
            },
            Effect::new(async move |tx| {
                while let Ok(res) = rx.recv_async().await {
                    _ = tx.send_async(Msg::EmuRes(res)).await;
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
                EmuResponse::Compiled(pc, decoded_ops) => {
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
                    self.no_effect()
                }
                EmuResponse::CpuUpdate(cpu) => {
                    self.emu_cpu = cpu;
                    self.no_effect()
                }
            },
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

                (self.dbg_page.decoded_ops_list, self.dbg_page.focus) = self
                    .dbg_page
                    .focus
                    .pipe(self.dbg_page.decoded_ops_list.update(&event));
                (self.dbg_page.cpu_gpr_list, self.dbg_page.focus) = self
                    .dbg_page
                    .focus
                    .pipe(self.dbg_page.cpu_gpr_list.update(&event));

                self.build_focus();

                self.no_effect()
            }
            _ => (self, Effect::none()),
        }
    }

    fn build_focus(&mut self) {
        self.dbg_page
            .focus
            .items(&self.dbg_page.decoded_ops_list)
            .next(&self.dbg_page.cpu_gpr_list)
            .commit();
    }

    async fn send_request(&self, req: EmuRequest) {
        _ = self.emu_handle.req_chan.0.send_async(req).await;
    }
}

async fn view(model: &Model) -> View {
    ui! {
        <Block .rounded .title="+ 🐷🎗️ P-ちゃん +" Width::grow() Height::grow() Direction::Horizontal Gap(1)>
            <Block Width::grow() Height::grow() MaxWidth::percentage(25)>
                <Instructions .model={model} Height::grow()/>
                <Summary .model={model}/>
            </Block>
            <Block>
                <CpuViewer .model={model} />
            </Block>
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
    let Some(instructions) = &model.dbg_page.decoded_ops else {
        return ui! {
            <Block .style={Style::new().dim()}>
                <Text>"Opcodes"</Text>
                " - 何も - Nothing here yet"
            </Block>
        };
    };

    ui! {
        <Block .rounded .title="+ mips dump +" Height::grow()>
            <ListViewCompact
                .state={&model.dbg_page.decoded_ops_list}
                .items={instructions.iter().cloned()}
                .highlight_style={Style::new().bg(Color::Green).fg(Color::Black)}
                Height::grow()
            />
        </Block>
    }
}

#[subview]
fn hseparator(style: Option<Style>) -> View {
    ui! {
        <Block
            .style={style.unwrap_or_default()}
            .borders={Borders::TOP}
            .border_type={BorderType::Plain}
            Width::grow() Height::fixed(1)
        />
    }
}

#[subview]
fn cpu_viewer(model: &Model) -> View {
    let show_zeroes = model.dbg_page.cpu_viewer_show_zeroes;
    let gpr = model
        .emu_cpu
        .gpr
        .iter()
        .enumerate()
        .filter(|(_, value)| show_zeroes || **value != 0)
        .map(|(reg, value)| {
            let style = match *value == 0 {
                true => Style::new().dim(),
                false => Style::default(),
            };
            let style = match reg.is_multiple_of(2) {
                true => style,
                false => style.fg(Color::from_u32(0xeeeeee)),
            };
            let reg = format!("${}", reg_str(reg as u8));
            let spacing = 8usize.saturating_sub(reg.len());
            Line::from_iter([
                Span::raw(" ".repeat(spacing)),
                Span::raw(reg),
                Span::raw(" "),
                Span::raw(hex(*value).to_string()),
            ])
            .style(style)
        });
    let pc = hex(model.emu_cpu.pc);
    ui! {
        <Block>
            <Block Width::grow()>
                "CPU View"
            </Block>
            <Block .rounded .title="+ gpr +" Padding::new(1, 2, 1, 1)>
                <Text>"     $pc {pc}"</Text>
                <ListViewCompact
                    .items={gpr}
                    .state={&model.dbg_page.cpu_gpr_list}
                    .highlight_style={Style::new().black().on_green().not_dim()}
                />
            </Block>
        </Block>
    }
}
