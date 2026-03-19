#[path = "./emu-task.rs"]
pub(crate) mod emu_task;

use color_eyre::Result;
use flume::{Receiver, Sender};
use manatui::prelude::*;
use manatui::ratatui::crossterm::event::Event;
use manatui::tea;
use manatui::tea::Effect;
use manatui::utils::keyv2;
use tokio::task::JoinHandle;

use crate::emu_task::{EmuRequest, EmuResponse, EmuTask, EmuTaskHandle};

pub(crate) type Chan<T> = (Sender<T>, Receiver<T>);

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tea::run()
        .init(Model::init)
        .view(view)
        .update(Model::update)
        .event_msg(Msg::Event)
        .quit_signal(|_, msg| matches!(msg, Msg::Quit))
        .run()
        .await?;

    Ok(())
}

struct Model {
    dbg_page:        DbgPage,
    emu_handle:      EmuTaskHandle,
    emu_join_handle: JoinHandle<Result<()>>,
}

struct DbgPage {}

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
                dbg_page: DbgPage {},
                emu_handle,
                emu_join_handle,
            },
            Effect::new(async move |tx| {
                while let Ok(res) = rx.recv_async().await {
                    _ = tx.send_async(Msg::EmuRes(res)).await;
                }
            }),
        )
    }

    async fn update(self, msg: Msg) -> (Self, Effect<Msg>) {
        match msg {
            Msg::Quit => unreachable!(),
            Msg::Event(event) => self.handle_event(event).await,
            Msg::EmuRes(res) => (self, Effect::none()),
        }
    }

    async fn handle_event(self, event: Event) -> (Self, Effect<Msg>) {
        match event {
            keyv2!(ctrl + 'c') => {
                self.send_request(EmuRequest::Quit).await;
                (self, Effect::msg(Msg::Quit))
            }
            _ => (self, Effect::none()),
        }
    }

    async fn send_request(&self, req: EmuRequest) {
        _ = self.emu_handle.req_chan.0.send_async(req).await;
    }
}

async fn view(model: &Model) -> View {
    ui! {
        "i am the magic rat 🐭🪄 (press ^C to exit.)"
    }
}
