pub(crate) mod utils;

use color_eyre::{eyre::Context, *};
use flume::{Receiver, Sender};
use mana_tui_elemental::prelude::*;
use mana_tui_macros::{subview, ui};
use pchan_emu::{
    Bus, Emu,
    bootloader::Bootloader,
    dynarec_v2::{FETCH_CHANNEL, PipelineV2, emitters::DecodedOp},
};
use pchan_utils::hex;
use std::{io::stdout, path::PathBuf};

use mana_tui_elemental::ui::View;
use mana_tui_potion::{Effect, Message, backends::DefaultEvent, focus::handlers::On};
use ratatui::{crossterm::event::KeyModifiers, style::Style};

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
    emu_chan:      (Sender<EmuRequest>, Receiver<EmuResponse>),
    compile_state: Option<CompileState>,
}

struct CompileState {
    pc:           u32,
    instructions: Vec<DecodedOp>,
}

#[derive(Debug, Clone)]
enum Msg {
    Quit,
    EmuRes(EmuResponse),
    EmuReq(EmuRequest),
}

impl Message for Msg {
    type Model = Model;
}

async fn init() -> (Model, Effect<Msg>) {
    let res_chan = flume::unbounded();
    let req_chan = flume::unbounded();
    let emu_task = EmuTask {
        rx: req_chan.1.clone(),
        tx: res_chan.0.clone(),
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
            emu_chan:      (req_chan.0, res_chan.1.clone()),
            compile_state: None,
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

async fn update(model: Model, msg: Msg) -> (Model, Effect<Msg>) {
    match msg {
        Msg::Quit => (model, Effect::none()),
        Msg::EmuRes(EmuResponse::Compiled(pc, instructions)) => (
            Model {
                compile_state: Some(CompileState { pc, instructions }),
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
    }
}

fn handle_quit(_: &Model, event: &DefaultEvent) -> Option<(Msg, Effect<Msg>)> {
    match event {
        key!(Char('q')) | key!(Char('c'), KeyModifiers::CONTROL) => {
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
            <Instructions .model={model} Width::grow() Height::grow()
                On::new(|_, event| {
                    match event {
                        key!(Char('n'))=>Some((Msg::EmuReq(EmuRequest::Step),Effect::none())),
                        _ => None
                    }
                })
            />
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
    let Some(CompileState { pc, instructions }) = &model.compile_state else {
        return ui! {
            <Block>
                <Text>"Instructions"</Text>
                <Hseparator .style={Style::new().dim()}/>
                "Nothing here yet"
            </Block>
        };
    };

    let instructions = instructions
        .iter()
        .enumerate()
        .map(|(idx, op)| format!("{} {op}", hex(idx as u32 * 4 + pc)));

    ui! {
        <Block>
            <Text>"Instructions"</Text>
            <Hseparator .style={Style::new().dim()}/>
            <List .items={instructions} Width::grow() Height::grow()></List>
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
}

#[derive(Debug, Clone)]
enum EmuResponse {
    Compiled(u32, Vec<DecodedOp>),
}

struct EmuTask {
    rx: Receiver<EmuRequest>,
    tx: Sender<EmuResponse>,
}

impl EmuTask {
    fn run(self) {
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

            while let Ok(msg) = self.rx.recv() {
                match msg {
                    EmuRequest::Step => {
                        pipe = pipe.step(&mut emu)?;
                    }
                    EmuRequest::Init(_) => {}
                }

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
                }
            }

            Ok(())
        });
    }
}
