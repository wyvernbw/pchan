use crate::{
    Emu,
    dynarec::{FetchParams, FetchSummary, TryCacheSummary},
    jit::{BlockFn, JIT},
};

pub struct EmuBundle {
    emu: Emu,
    jit: JIT,
}

pub enum EmuDynarecPipeline {
    Uninit,
    Init { pc: u32 },
    TriedCache { pc: u32, result: TryCacheSummary },
    Fetched { pc: u32, fetch: FetchSummary },
    Emitted { pc: u32, function: BlockFn },
    Called { pc: u32, function: BlockFn },
    Cached,
}

impl EmuDynarecPipeline {
    pub const fn new() -> Self {
        Self::Uninit
    }
    pub const fn from_emu(emu: &Emu) -> Self {
        Self::Init { pc: emu.cpu.pc }
    }
    pub const fn as_fetch(&self) -> Option<&FetchSummary> {
        match self {
            EmuDynarecPipeline::Fetched { fetch, .. } => Some(fetch),
            _ => None,
        }
    }
    pub fn try_init(self, pc: u32) -> Self {
        if let Self::Uninit = self {
            Self::Init { pc }
        } else {
            self
        }
    }
    pub fn step(self, emu: &mut Emu, jit: &mut JIT) -> color_eyre::Result<Self> {
        match self {
            EmuDynarecPipeline::Uninit => Ok(EmuDynarecPipeline::Uninit),
            EmuDynarecPipeline::Init { pc } => {
                let call = emu.try_cache_call(pc);
                Ok(EmuDynarecPipeline::TriedCache { pc, result: call })
            }
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Success,
                ..
            } => Ok(EmuDynarecPipeline::from_emu(emu)),
            EmuDynarecPipeline::TriedCache {
                pc,
                result: TryCacheSummary::Fail,
            } => {
                let fetch_result = {
                    emu.fetch(FetchParams::builder().pc(emu.cpu.pc).build())
                        .unwrap()
                };
                Ok(Self::Fetched {
                    pc,
                    fetch: fetch_result,
                })
            }
            EmuDynarecPipeline::Fetched { pc, fetch } => {
                let ptr_type = jit.pointer_type();
                let (func_id, mut func) = jit.create_function(pc)?;
                let func_ref_table = jit.create_func_ref_table(&mut func);
                let mut fn_builder = jit.create_fn_builder(&mut func);

                emu.emit_function()
                    .ptr_type(ptr_type)
                    .fetch_result(fetch)
                    .fn_builder(&mut fn_builder)
                    .func_ref_table(&func_ref_table)
                    .call();

                Emu::destroy_fn_builder(fn_builder);
                let function = jit.finish_function(func_id, func.clone())?;

                tracing::info!("{:?} fn compiled", function.fn_ptr);

                Ok(Self::Emitted { pc, function })
            }
            EmuDynarecPipeline::Emitted { pc, function } => {
                function(emu, true);
                Ok(Self::Called { pc, function })
            }
            EmuDynarecPipeline::Called { pc, function } => {
                emu.jit_cache.fn_map.insert(pc, function);
                Ok(Self::Cached)
            }
            EmuDynarecPipeline::Cached => Ok(Self::from_emu(emu)),
        }
    }
    pub fn run(mut self, emu: &mut Emu, jit: &mut JIT) -> color_eyre::Result<Self> {
        loop {
            self = self.step(emu, jit)?;
        }
    }
    pub const fn progress(&self) -> usize {
        match self {
            EmuDynarecPipeline::Uninit => 0,
            EmuDynarecPipeline::Init { .. } => 0,
            EmuDynarecPipeline::TriedCache { .. } => 1,
            EmuDynarecPipeline::Fetched { .. } => 2,
            EmuDynarecPipeline::Emitted { .. } => 3,
            EmuDynarecPipeline::Called { .. } => 4,
            EmuDynarecPipeline::Cached => 5,
        }
    }
    pub const fn max_progress() -> usize {
        5
    }
}

impl Default for EmuDynarecPipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EmuDynarecPipelineReport {
    pub current: &'static str,
    pub next: &'static str,
    pub progress: usize,
}

impl EmuDynarecPipelineReport {
    pub const fn not_started() -> EmuDynarecPipelineReport {
        EmuDynarecPipelineReport {
            current: "not started",
            next: "init",
            progress: 0,
        }
    }
}

impl EmuDynarecPipeline {
    pub const fn report(&self) -> EmuDynarecPipelineReport {
        let progress = self.progress();
        match self {
            EmuDynarecPipeline::Uninit => EmuDynarecPipelineReport::not_started(),
            EmuDynarecPipeline::Init { .. } => EmuDynarecPipelineReport {
                progress,
                current: "ready",
                next: "try cache",
            },
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Success,
                ..
            } => EmuDynarecPipelineReport {
                progress,
                current: "cache success",
                next: "restart",
            },
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Fail,
                ..
            } => EmuDynarecPipelineReport {
                progress,
                current: "cache fail",
                next: "fetch",
            },
            EmuDynarecPipeline::Fetched { .. } => EmuDynarecPipelineReport {
                progress,
                current: "fetched",
                next: "emit",
            },
            EmuDynarecPipeline::Emitted { .. } => EmuDynarecPipelineReport {
                progress,
                current: "emitted",
                next: "call",
            },
            EmuDynarecPipeline::Called { .. } => EmuDynarecPipelineReport {
                progress,
                current: "called",
                next: "cache",
            },
            EmuDynarecPipeline::Cached => EmuDynarecPipelineReport {
                progress,
                current: "cached",
                next: "restart",
            },
        }
    }
}
