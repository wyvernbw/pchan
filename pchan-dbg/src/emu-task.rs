use crate::Chan;
use color_eyre::{Result, eyre::Context};
use pchan_emu::{
    Bus, Emu,
    bootloader::Bootloader,
    dynarec_v2::{DynarecBlock, PipelineV2, PipelineV2Stage},
    gpu::Gpu,
};
use pchan_gpu::Renderer;
use pchan_utils::hex;
use std::{
    collections::{HashSet, VecDeque},
    path::PathBuf,
    sync::Arc,
};
use tokio::{task::JoinHandle, time::Instant};
use tracing::instrument;

#[derive(Debug, Clone)]
pub(crate) enum EmuRequest {
    Quit,
    Step,
    Run,
    Pause,
    AddBreakpoint(u32),
    DelBreakpoint(u32),
    HardReset,
}
#[derive(Debug, Clone)]
pub(crate) enum EmuResponse {
    StageUpdate(PipelineV2Stage),
    StateUpdate(EmuTaskState),
    Compiled(DynarecBlock),
    ObjDump(Arc<str>),
    FrequencyUpdate(f64),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmuTaskState {
    Running,
    Paused,
    Done,
}
pub(crate) struct EmuTask {
    handle:        EmuTaskHandle,
    state:         EmuTaskState,
    bios_path:     PathBuf,
    emu:           &'static mut Emu,
    pipe:          PipelineV2,
    cycle_samples: VecDeque<(u64, Instant)>,
    breakpoints:   HashSet<u32>,
}
#[derive(Debug, Clone)]
pub(crate) struct EmuTaskHandle {
    pub(crate) req_chan: Chan<EmuRequest>,
    pub(crate) res_chan: Chan<EmuResponse>,
    pub(crate) dbg_view: DebugView,
    pub(crate) tty_chan: Chan<Arc<str>>,
}

impl EmuTask {
    pub(crate) async fn spawn() -> Result<(EmuTaskHandle, std::thread::JoinHandle<Result<()>>)> {
        let req_chan = flume::unbounded();
        let res_chan = flume::unbounded();
        let bios_path = std::env::var("PCHAN_BIOS")
            .wrap_err("PCHAN_BIOS env var not set.")?
            .parse::<PathBuf>()
            .wrap_err("PCHAN_BIOS var contains invalid path.")?;

        let emu = Box::leak(Box::new(Emu::default()));
        let mut renderer = Renderer::try_new().await?;
        let tty_chan = emu.tty.set_channeled();
        renderer.connect_emu(emu);
        renderer.start();
        let handle = EmuTaskHandle {
            req_chan,
            res_chan,
            dbg_view: DebugView::from_emu(emu),
            tty_chan,
        };
        let task = EmuTask {
            handle: handle.clone(),
            state: EmuTaskState::Paused,
            emu,
            pipe: PipelineV2::Uninit,
            bios_path,
            cycle_samples: VecDeque::with_capacity(256),
            breakpoints: HashSet::new(),
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
    fn hard_reset(&mut self) -> Result<()> {
        let mut new_emu = Emu::default();
        new_emu.gpu_reconnect(self.emu);
        *self.emu = new_emu;
        self.emu
            .tty
            .set_channeled_with(self.handle.tty_chan.0.clone());
        self.emu.set_bios_path(&self.bios_path);
        self.emu.load_bios()?;
        self.pipe = PipelineV2::new(self.emu);
        self.soft_reset();
        Ok(())
    }
    fn soft_reset(&mut self) {
        self.emu.cpu_mut().jump_to_bios();
    }
    fn run(mut self) -> std::thread::JoinHandle<Result<()>> {
        std::thread::spawn(move || -> Result<()> {
            self.hard_reset()?;
            let stage = self.pipe.stage();
            self.handle
                .res_chan
                .0
                .send(EmuResponse::StageUpdate(stage))?;
            loop {
                match self.state {
                    EmuTaskState::Paused => {
                        if let Ok(msg) = self.handle.req_chan.1.recv() {
                            self = self.handle_req(msg)?;
                            self.handle_pipe_effects()?;
                        }
                        let stage = self.pipe.stage();
                        self.handle
                            .res_chan
                            .0
                            .send(EmuResponse::StageUpdate(stage))?;
                    }
                    EmuTaskState::Running => {
                        if let Ok(msg) = self.handle.req_chan.1.try_recv() {
                            self = self.handle_req(msg)?;
                        }

                        self.pipe = self.pipe.step(self.emu)?;
                        self.handle_pipe_effects()?;
                    }
                    EmuTaskState::Done => return Ok(()),
                }
            }
        })
    }

    fn try_send_res(&self, res: EmuResponse) {
        if self.handle.res_chan.1.is_full() {
            return;
        }
        _ = self.handle.res_chan.0.send(res);
    }

    fn cycles_since_last_sample(&self, clock: u64) -> Option<u64> {
        self.cycle_samples
            .back()
            .map(|(x, _)| clock.saturating_sub(*x))
    }

    fn handle_pipe_effects(&mut self) -> Result<()> {
        let cycles_since_last_sample = self
            .cycles_since_last_sample(self.emu.cpu.cycles)
            .unwrap_or(u64::MAX);
        if cycles_since_last_sample >= 8_000_000 {
            let sample = (self.emu.cpu().cycles, Instant::now());
            const MAX_SAMPLES: usize = 256;
            self.cycle_samples.push_back(sample);
            if self.cycle_samples.len() > MAX_SAMPLES {
                self.cycle_samples.pop_front();
            }
            if self.cycle_samples.len() > 1
                && let Some((first, last)) =
                    self.cycle_samples.front().zip(self.cycle_samples.back())
            {
                let (cycle_delta, overflowed) = last.0.overflowing_sub(first.0);
                if overflowed {
                    self.cycle_samples.clear();
                    return Ok(());
                }
                let time_delta = last.1.duration_since(first.1);
                let frequency_hz = cycle_delta as f64 / time_delta.as_secs_f64();
                self.try_send_res(EmuResponse::FrequencyUpdate(frequency_hz));
            }
        }

        match &self.pipe {
            PipelineV2::Uninit => {}
            PipelineV2::Init { dynarec, pc } => {}
            PipelineV2::Compiled { func, .. } => {
                if self.state != EmuTaskState::Running || (cycles_since_last_sample >= 8_000_000) {
                    self.handle
                        .res_chan
                        .0
                        .send(EmuResponse::Compiled(func.clone()))?;
                }

                if self.state == EmuTaskState::Running
                    && self.breakpoints.contains(&self.emu.cpu().pc)
                {
                    self.state = EmuTaskState::Paused;
                }
            }
            PipelineV2::Called { .. } => {}
            PipelineV2::Cached { dynarec, scheduler } => {}
        };
        Ok(())
    }
    #[instrument(skip(self))]
    fn handle_req(mut self, req: EmuRequest) -> Result<Self> {
        match req {
            EmuRequest::Step => {
                self.pipe = self.pipe.step(self.emu)?;
                let stage = self.pipe.stage();
                self.handle
                    .res_chan
                    .0
                    .send(EmuResponse::StageUpdate(stage))?;
            }
            EmuRequest::Quit => {
                self.state = EmuTaskState::Done;
            }
            EmuRequest::Run => {
                self = self.handle_req(EmuRequest::Step)?; // skip past potential breakpoint
                self.state = EmuTaskState::Running;
                self.handle
                    .res_chan
                    .0
                    .send(EmuResponse::StateUpdate(self.state))?;
            }
            EmuRequest::Pause => {
                self.state = EmuTaskState::Paused;
                self.handle
                    .res_chan
                    .0
                    .send(EmuResponse::StateUpdate(self.state))?;
                let stage = self.pipe.stage();
                _ = self.handle.res_chan.0.send(EmuResponse::StageUpdate(stage));
            }
            EmuRequest::AddBreakpoint(addr) => {
                tracing::debug!("brk added {}", hex(addr));
                self.breakpoints.insert(addr);
            }
            EmuRequest::DelBreakpoint(addr) => {
                tracing::debug!("brk deleted {}", hex(addr));
                self.breakpoints.remove(&addr);
            }
            EmuRequest::HardReset => {
                tracing::info!("hard reset");
                self.hard_reset()?;
                let stage = self.pipe.stage();
                self.state = EmuTaskState::Paused;
                self.handle
                    .res_chan
                    .0
                    .send(EmuResponse::StageUpdate(stage))?;
                self.handle
                    .res_chan
                    .0
                    .send(EmuResponse::StateUpdate(self.state))?;
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

impl Default for DebugView {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}
