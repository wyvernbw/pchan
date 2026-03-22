use crate::Chan;
use color_eyre::{Result, eyre::Context};
use flume::Receiver;
use pchan_emu::{
    Bus, Emu,
    bootloader::Bootloader,
    cpu::Cpu,
    dynarec_v2::{DynarecBlock, FETCH_CHANNEL, PipelineV2, PipelineV2Stage, emitters::DecodedOp},
};
use std::{io::Write, path::PathBuf, process::Stdio, sync::Arc};
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub(crate) enum EmuRequest {
    Quit,
    Step,
}
#[derive(Debug, Clone)]
pub(crate) enum EmuResponse {
    StageUpdate(PipelineV2Stage),
    StateUpdate(EmuTaskState),
    Compiled(u32, Arc<[DecodedOp]>, DynarecBlock),
    CpuUpdate(Box<Cpu>),
    ObjDump(Arc<str>),
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum EmuTaskState {
    Running,
    Paused,
    Done,
}
pub(crate) struct EmuTask {
    handle:    EmuTaskHandle,
    state:     EmuTaskState,
    bios_path: PathBuf,
    emu:       &'static mut Emu,
    pipe:      PipelineV2,
}
#[derive(Debug, Clone)]
pub(crate) struct EmuTaskHandle {
    pub(crate) req_chan: Chan<EmuRequest>,
    pub(crate) res_chan: Chan<EmuResponse>,
    pub(crate) dbg_view: DebugView,
    pub(crate) tty_rx:   Receiver<Arc<str>>,
}

impl EmuTask {
    pub(crate) fn spawn() -> Result<(EmuTaskHandle, JoinHandle<Result<()>>)> {
        let req_chan = flume::unbounded();
        let res_chan = flume::unbounded();
        let bios_path = std::env::var("PCHAN_BIOS")
            .wrap_err("PCHAN_BIOS env var not set.")?
            .parse::<PathBuf>()
            .wrap_err("PCHAN_BIOS var contains invalid path.")?;

        let emu = Box::leak(Box::new(Emu::default()));
        let tty_rx = emu.tty.set_channeled();
        let handle = EmuTaskHandle {
            req_chan,
            res_chan,
            dbg_view: DebugView::from_emu(emu),
            tty_rx,
        };
        let task = EmuTask {
            handle: handle.clone(),
            state: EmuTaskState::Paused,
            emu,
            pipe: PipelineV2::Uninit,
            bios_path,
        };
        let join_handle = task.run();

        Ok((handle, join_handle))
    }
    pub(crate) fn handle(&self) -> &EmuTaskHandle {
        &self.handle
    }
    pub(crate) fn clone_handle(&self) -> EmuTaskHandle {
        self.handle().clone()
    }
    fn run(mut self) -> JoinHandle<Result<()>> {
        tokio::task::spawn_blocking(move || -> Result<()> {
            self.emu.set_bios_path(&self.bios_path);
            self.emu.load_bios()?;
            self.emu.cpu_mut().jump_to_bios();

            self.pipe = PipelineV2::new(self.emu);

            loop {
                match self.state {
                    EmuTaskState::Paused => {
                        if let Ok(msg) = self.handle.req_chan.1.recv() {
                            self = self.handle_req(msg)?;
                            self.handle_pipe_effects()?;
                        }
                    }
                    EmuTaskState::Running => loop {
                        if let Ok(msg) = self.handle.req_chan.1.try_recv() {
                            self = self.handle_req(msg)?;
                        }

                        self.pipe = self.pipe.step(self.emu)?;
                        let stage = self.pipe.stage();
                        self.handle
                            .res_chan
                            .0
                            .send(EmuResponse::StageUpdate(stage))?;

                        self.handle_pipe_effects()?;
                    },
                    EmuTaskState::Done => return Ok(()),
                }
            }
        })
    }
    fn handle_pipe_effects(&mut self) -> Result<()> {
        match &self.pipe {
            PipelineV2::Uninit => {}
            PipelineV2::Init { dynarec, pc } => {}
            PipelineV2::Compiled { func, .. } => {
                let mut first_addr = None;
                let instructions = FETCH_CHANNEL
                    .1
                    .try_iter()
                    .map(|(addr, op)| {
                        first_addr = first_addr.or(Some(addr));
                        op
                    })
                    .collect();
                if let Some(first_addr) = first_addr {
                    self.handle.res_chan.0.send(EmuResponse::Compiled(
                        first_addr,
                        instructions,
                        func.clone(),
                    ))?;
                };
            }
            PipelineV2::Called {
                pc,
                times,
                func,
                dynarec,
                scheduler,
            } => {
                let cpu = Box::clone_from_ref(self.emu.cpu());
                self.emu.mem_mut();
                self.handle.res_chan.0.send(EmuResponse::CpuUpdate(cpu))?;
            }
            PipelineV2::Cached { dynarec, scheduler } => {}
        };
        Ok(())
    }
    fn handle_req(mut self, req: EmuRequest) -> Result<Self> {
        match req {
            EmuRequest::Step => {
                self.pipe = self.pipe.step(&mut self.emu)?;
                let stage = self.pipe.stage();
                self.handle
                    .res_chan
                    .0
                    .send(EmuResponse::StageUpdate(stage))?;
            }
            EmuRequest::Quit => {
                self.state = EmuTaskState::Done;
            }
        };
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DebugView(*mut Emu);

unsafe impl Send for DebugView {}
unsafe impl Sync for DebugView {}

impl DebugView {
    pub fn from_emu(emu: &mut Emu) -> Self {
        Self(emu as *mut Emu)
    }

    pub fn emu(&self) -> &Emu {
        unsafe { &*self.0 }
    }
}
