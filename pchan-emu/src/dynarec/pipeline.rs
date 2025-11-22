use crate::{
    Emu,
    dynarec::{FetchParams, FetchSummary},
    jit::{BlockFn, JIT},
};

pub struct EmuBundle {
    emu: Emu,
    jit: JIT,
}

pub enum EmuDynarecPipeline {
    Init,
    Fetched { fetch: FetchSummary },
    Emitted { function: BlockFn },
    Called { function: BlockFn },
    Cached,
}

impl EmuDynarecPipeline {
    pub const fn new() -> Self {
        Self::Init
    }
    pub const fn as_fetch(&self) -> Option<&FetchSummary> {
        match self {
            EmuDynarecPipeline::Fetched { fetch } => Some(fetch),
            _ => None,
        }
    }
    pub fn step(self, emu: &mut Emu, jit: &mut JIT, pc: u32) -> color_eyre::Result<Self> {
        match self {
            EmuDynarecPipeline::Init => {
                let fetch_result = {
                    emu.fetch(FetchParams::builder().pc(emu.cpu.pc).build())
                        .unwrap()
                };
                Ok(Self::Fetched {
                    fetch: fetch_result,
                })
            }
            EmuDynarecPipeline::Fetched { fetch } => {
                let ptr_type = jit.pointer_type();
                let (func_id, mut func) = jit.create_function(emu.cpu.pc)?;
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

                Ok(Self::Emitted { function })
            }
            EmuDynarecPipeline::Emitted { function } => {
                function(emu, true);
                Ok(Self::Called { function })
            }
            EmuDynarecPipeline::Called { function } => {
                emu.jit_cache.fn_map.insert(pc, function);
                Ok(Self::Cached)
            }
            EmuDynarecPipeline::Cached => Ok(Self::Init),
        }
    }
    pub fn run(mut self, emu: &mut Emu, jit: &mut JIT, pc: u32) -> color_eyre::Result<Self> {
        loop {
            self = self.step(emu, jit, pc)?;
        }
    }
    pub const fn progress(&self) -> usize {
        match self {
            EmuDynarecPipeline::Init => 0,
            EmuDynarecPipeline::Fetched { .. } => 1,
            EmuDynarecPipeline::Emitted { .. } => 2,
            EmuDynarecPipeline::Called { .. } => 3,
            EmuDynarecPipeline::Cached => 4,
        }
    }
    pub const fn max_progress() -> usize {
        4
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

impl EmuDynarecPipeline {
    pub const fn report(&self) -> EmuDynarecPipelineReport {
        match self {
            EmuDynarecPipeline::Init => EmuDynarecPipelineReport {
                progress: 0,
                current: "ready",
                next: "fetch",
            },
            EmuDynarecPipeline::Fetched { .. } => EmuDynarecPipelineReport {
                progress: 1,
                current: "fetched",
                next: "emit",
            },
            EmuDynarecPipeline::Emitted { .. } => EmuDynarecPipelineReport {
                progress: 2,
                current: "emitted",
                next: "call",
            },
            EmuDynarecPipeline::Called { .. } => EmuDynarecPipelineReport {
                progress: 3,
                current: "called",
                next: "cache",
            },
            EmuDynarecPipeline::Cached => EmuDynarecPipelineReport {
                progress: 4,
                current: "cached",
                next: "restart",
            },
        }
    }
}
