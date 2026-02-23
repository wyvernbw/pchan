pub(crate) mod utils;

use color_eyre::{eyre::Context, *};
use flume::{Receiver, Sender};
use mana_tui::prelude::strum::EnumCount;
use mana_tui_elemental::prelude::*;
use mana_tui_macros::{subview, ui};
use pchan_emu::{
    Bus, Emu,
    bootloader::Bootloader,
    dynarec_v2::{FETCH_CHANNEL, PipelineV2, PipelineV2Stage, emitters::DecodedOp},
};
use pchan_utils::hex;
use std::{
    io::stdout,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use mana_tui_elemental::layout::IntoStateful;
use mana_tui_elemental::ui::View;
use mana_tui_potion::{Effect, Message, backends::DefaultEvent, focus::handlers::On};
use ratatui::{
    crossterm::event::KeyModifiers,
    style::{Color, Style},
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;
    fn should_quit(_: &Model, msg: &Msg) -> bool {
        matches!(msg, Msg::Quit)
    }

    mana_tui_potion::run()
        .quit_signal(should_quit)
        .writer(stdout())
        .init(init)
        .update(update)
        .view(view)
        .run()
        .await?;

    Ok(())
}

struct Model {
    emu_chan:           (Sender<EmuRequest>, Receiver<EmuResponse>),
    compile_state:      Option<Arc<[Text<'static>]>>,
    // a little bit of `Arc<Mutex>` in my life
    compile_list_state: Arc<Mutex<ListState>>,
    emu_stage:          PipelineV2Stage,
    emu_task_state:     EmuTaskState,
}

#[derive(Debug, Clone)]
enum NavMsg {
    Up,
    Down,
}

#[derive(Debug, Clone)]
enum EmuTaskState {
    Paused,
    Running,
    Done,
}

#[derive(Debug, Clone)]
enum Msg {
    Quit,
    EmuRes(EmuResponse),
    EmuReq(EmuRequest),
    InstructionsNav(NavMsg),
}

impl Message for Msg {
    type Model = Model;
}

async fn init() -> (Model, Effect<Msg>) {
    let res_chan = flume::unbounded();
    let req_chan = flume::unbounded();
    let emu_task = EmuTask {
        rx:    req_chan.1.clone(),
        tx:    res_chan.0.clone(),
        state: EmuTaskState::Paused,
    };

    // TODO: proper bios path setup flow
    req_chan
        .0
        .send(EmuRequest::Init(EmuConfig {
            bios_path: std::env::var("PCHAN_BIOS")
                .wrap_err("Failed to read BIOS path from env var")
                .unwrap()
                .parse()
                .wrap_err("PCHAN_BIOS env variable must hold a valid path")
                .unwrap(),
        }))
        .unwrap();

    emu_task.run();
    (
        Model {
            emu_chan:           (req_chan.0, res_chan.1.clone()),
            compile_state:      None,
            compile_list_state: Arc::new(Mutex::new(ListState::default())),
            emu_stage:          PipelineV2Stage::Uninit,
            emu_task_state:     EmuTaskState::Paused,
        },
        Effect::new(move |tx| {
            let rx = res_chan.1.clone();
            async move {
                while let Ok(res) = rx.recv_async().await {
                    _ = tx.send(Msg::EmuRes(res));
                }
            }
        }),
    )
}

async fn update(mut model: Model, msg: Msg) -> (Model, Effect<Msg>) {
    match msg {
        Msg::Quit => unreachable!(),
        Msg::EmuRes(EmuResponse::Compiled(pc, instructions)) => {
            let mut compile_list_state = ListState::default();
            compile_list_state.select_first();
            let compile_list_state = Arc::new(Mutex::new(compile_list_state));
            let instructions = instructions
                .iter()
                .enumerate()
                .map(|(idx, op)| format!("{} {op}", hex(idx as u32 * 4 + pc)).into())
                .collect();
            (
                Model {
                    compile_state: Some(instructions),
                    compile_list_state,
                    ..model
                },
                Effect::none(),
            )
        }
        Msg::EmuRes(EmuResponse::StageUpdate(stage)) => (
            Model {
                emu_stage: stage,
                ..model
            },
            Effect::none(),
        ),
        Msg::EmuRes(EmuResponse::StateUpdate(state)) => (
            Model {
                emu_task_state: state,
                ..model
            },
            Effect::none(),
        ),
        Msg::EmuReq(emu_request) => {
            model
                .emu_chan
                .0
                .send(emu_request)
                .expect("failed to send message!");
            (model, Effect::none())
        }
        Msg::InstructionsNav(NavMsg::Up) => {
            model.compile_list_state.lock().unwrap().select_previous();
            (model, Effect::none())
        }
        Msg::InstructionsNav(NavMsg::Down) => {
            // if model.compile_list_state.selected()
            //     != model
            //         .compile_state
            //         .as_ref()
            //         .map(|list| list.len().saturating_sub(1))
            {
                model.compile_list_state.lock().unwrap().select_next();
            }
            (model, Effect::none())
        }
    }
}

fn handle_quit(model: &Model, event: &DefaultEvent) -> Option<(Msg, Effect<Msg>)> {
    match event {
        key!(Char('q')) | key!(Char('c'), KeyModifiers::CONTROL) => {
            model
                .emu_chan
                .0
                .send(EmuRequest::Quit)
                .expect("failed to kill emu thread");
            Some((Msg::Quit, Effect::none()))
        }
        _ => None,
    }
}

async fn view(model: &Model) -> View {
    ui! {
        <Block .rounded .title_top={"+ 🐷🎀 P-ちゃん dbg +"} On::new(handle_quit) Width::grow() Height::grow()
            Direction::Horizontal
        >
            <Block Width::grow() Height::grow()>
                <Instructions .model={model} Width::grow() Height::grow()
                    Padding::uniform(1)
                    On::new(|_, event| {
                        match event {
                            key!(Char('n'))=>Some((Msg::EmuReq(EmuRequest::Step),Effect::none())),
                            key!(Char('r'))=>Some((Msg::EmuReq(EmuRequest::Run),Effect::none())),
                            key!(Char('p'))=>Some((Msg::EmuReq(EmuRequest::Pause),Effect::none())),
                            _ => None
                        }
                    })
                />
                <Summary .model={model} Height::fixed(5) Width::grow() />
            </Block>
            <Block Width::grow() Height::grow()></Block>
            <Block Width::grow() Height::grow()></Block>
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
fn instructions(model: &Model) -> View {
    let Some(instructions) = &model.compile_state else {
        return ui! {
            <Block>
                <Text>"Instructions"</Text>
                <Hseparator .style={Style::new().dim()}/>
                "Nothing here yet"
            </Block>
        };
    };

    ui! {
        <Block>
            <Text>"Instructions"</Text>
            <Hseparator .style={Style::new().dim()}/>
            <List
                .items={instructions.iter().cloned()}
                .highlight_style={Style::new().bg(Color::Green).fg(Color::Black)}
                .stateful
                {On::<Msg>::new(|_, event| {
                    match event {
                        key!(Char('j')) => Some((Msg::InstructionsNav(NavMsg::Down), Effect::none())),
                        key!(Char('k')) => Some((Msg::InstructionsNav(NavMsg::Up), Effect::none())),
                        _ => None
                    }
                })}
                {model.compile_list_state.clone()}
                Width::grow() Height::grow()
            />
        </Block>
    }
}

#[subview]
fn summary(model: &Model) -> View {
    let stage = model.emu_stage;
    let stage_idx = (model.emu_stage as u8).saturating_sub(1);
    let max_stage_idx = PipelineV2Stage::COUNT as u8 - 2;
    let ratio = stage_idx as f64 / max_stage_idx as f64;
    let emu_task_state = &model.emu_task_state;
    ui! {
        <Block .rounded {Padding::new(2, 2, 1, 1)}>
            <Block
                Direction::Horizontal Gap(1) MainJustify::SpaceBetween Width::grow()
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

            <Text>"{emu_task_state:?}"</Text>
        </Block>
    }
}

#[derive(Debug, Clone)]
struct EmuConfig {
    bios_path: PathBuf,
}

#[derive(Debug, Clone)]
enum EmuRequest {
    Init(EmuConfig),
    Step,
    Run,
    Pause,
    Quit,
}

#[derive(Debug, Clone)]
enum EmuResponse {
    StageUpdate(PipelineV2Stage),
    StateUpdate(EmuTaskState),
    Compiled(u32, Vec<DecodedOp>),
}

struct EmuTask {
    rx:    Receiver<EmuRequest>,
    tx:    Sender<EmuResponse>,
    state: EmuTaskState,
}

impl EmuTask {
    fn update(&mut self, state: EmuTaskState) -> Result<()> {
        self.state = state.clone();
        self.tx.send(EmuResponse::StateUpdate(state))?;
        Ok(())
    }
    fn handle_req(
        &mut self,
        emu: &mut Emu,
        mut pipe: PipelineV2,
        req: EmuRequest,
    ) -> Result<PipelineV2> {
        match req {
            EmuRequest::Step => {
                pipe = pipe.step(emu)?;
                let stage = pipe.stage();
                self.tx.send(EmuResponse::StageUpdate(stage))?;
            }
            EmuRequest::Init(_) => {}
            EmuRequest::Run => {
                self.update(EmuTaskState::Running)?;
            }
            EmuRequest::Pause => {
                self.update(EmuTaskState::Paused)?;
            }
            EmuRequest::Quit => {
                self.update(EmuTaskState::Done)?;
            }
        };
        Ok(pipe)
    }
    fn handle_pipe_effects(&mut self, pipe: &PipelineV2) -> Result<()> {
        match &pipe {
            PipelineV2::Uninit => {}
            PipelineV2::Init { dynarec, pc } => {}
            PipelineV2::Compiled { .. } => {
                let mut first_addr = None;
                let instructions = FETCH_CHANNEL
                    .1
                    .try_iter()
                    .map(|(addr, op)| {
                        first_addr = first_addr.or(Some(addr));
                        op
                    })
                    .collect::<Vec<_>>();
                if let Some(first_addr) = first_addr {
                    self.tx
                        .send(EmuResponse::Compiled(first_addr, instructions))?;
                };
            }
            PipelineV2::Called {
                pc,
                times,
                func,
                dynarec,
                scheduler,
            } => {}
            PipelineV2::Cached { dynarec, scheduler } => {}
        };
        Ok(())
    }
    fn run(mut self) {
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut emu = Emu::default();
            let mut pipe = PipelineV2::new(&emu);

            loop {
                let Ok(req) = self.rx.recv() else {
                    return Ok(());
                };
                let EmuRequest::Init(config) = req else {
                    continue;
                };

                emu.set_bios_path(config.bios_path);
                emu.load_bios()?;
                emu.cpu_mut().jump_to_bios();

                break;
            }

            loop {
                match self.state {
                    EmuTaskState::Paused => {
                        if let Ok(msg) = self.rx.recv() {
                            pipe = self.handle_req(&mut emu, pipe, msg)?;

                            self.handle_pipe_effects(&pipe)?;
                        }
                    }
                    EmuTaskState::Running => loop {
                        if let Ok(msg) = self.rx.try_recv() {
                            pipe = self.handle_req(&mut emu, pipe, msg)?;
                        }

                        pipe = pipe.step(&mut emu)?;
                        let stage = pipe.stage();
                        self.tx.send(EmuResponse::StageUpdate(stage))?;

                        self.handle_pipe_effects(&pipe)?;
                    },
                    EmuTaskState::Done => return Ok(()),
                }
            }
        });
    }
}
