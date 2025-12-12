use std::time::{Duration, Instant};

use crate::{
    Emu,
    dynarec::{FetchParams, FetchSummary, TryCacheSummary},
    jit::{BlockFn, JIT},
};

pub struct EmuBundle {
    emu: Emu,
    jit: JIT,
}

#[derive(Debug, Clone)]
pub enum EmuDynarecPipeline {
    Uninit,
    Init {
        pc: u32,
    },
    Fetched {
        elapsed: Duration,
        pc: u32,
        fetch: FetchSummary,
    },
    TriedCache {
        elapsed: Duration,
        pc: u32,
        fetch: FetchSummary,
        result: TryCacheSummary,
    },
    Emitted {
        elapsed: Duration,
        pc: u32,
        function: BlockFn,
    },
    Called {
        elapsed: Duration,
        pc: u32,
        function: BlockFn,
    },
    Cached {
        elapsed: Duration,
    },
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
                let now = Instant::now();
                let fetch_result = match emu.inst_cache.get(pc).cloned() {
                    None => {
                        let fetch_result =
                            emu.fetch(FetchParams::builder().pc(pc).build()).unwrap();
                        emu.inst_cache.insert(pc, fetch_result.clone());
                        fetch_result
                    }
                    Some(fetch_result) => fetch_result,
                };
                Ok(Self::Fetched {
                    elapsed: now.elapsed(),
                    pc,
                    fetch: fetch_result,
                })
            }
            EmuDynarecPipeline::Fetched {
                pc,
                fetch,
                elapsed: _,
            } => {
                let now = Instant::now();
                let call = emu.try_cache_call(pc, &fetch.decoded_ops);
                Ok(EmuDynarecPipeline::TriedCache {
                    elapsed: now.elapsed(),
                    pc,
                    fetch,
                    result: call,
                })
            }
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Success,
                ..
            } => Ok(EmuDynarecPipeline::from_emu(emu)),
            EmuDynarecPipeline::TriedCache {
                pc,
                result: TryCacheSummary::Fail(hash),
                fetch,
                elapsed: _,
            } => {
                let now = Instant::now();
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
                let function = jit.finish_function(func_id, func.clone(), hash)?;

                tracing::info!("{:?} fn compiled", function.fn_ptr);

                Ok(Self::Emitted {
                    pc,
                    function,
                    elapsed: now.elapsed(),
                })
            }
            EmuDynarecPipeline::Emitted {
                pc,
                function,
                elapsed: _,
            } => {
                let now = Instant::now();
                function(emu, false);
                Ok(Self::Called {
                    pc,
                    function,
                    elapsed: now.elapsed(),
                })
            }
            EmuDynarecPipeline::Called {
                pc,
                function,
                elapsed: _,
            } => {
                let now = Instant::now();
                emu.jit_cache.fn_map.insert(pc, function);
                Ok(Self::Cached {
                    elapsed: now.elapsed(),
                })
            }
            EmuDynarecPipeline::Cached { .. } => Ok(Self::from_emu(emu)),
        }
    }
    pub fn run_once(mut self, emu: &mut Emu, jit: &mut JIT) -> color_eyre::Result<Self> {
        while self.progress() != Self::max_progress() {
            self = self.step(emu, jit)?;
        }
        Ok(self)
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
            EmuDynarecPipeline::Fetched { .. } => 1,
            EmuDynarecPipeline::TriedCache { .. } => 2,
            EmuDynarecPipeline::Emitted { .. } => 3,
            EmuDynarecPipeline::Called { .. } => 4,
            EmuDynarecPipeline::Cached { .. } => 5,
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
    pub elapsed: Duration,
}

impl EmuDynarecPipelineReport {
    pub const fn not_started() -> EmuDynarecPipelineReport {
        EmuDynarecPipelineReport {
            current: "not started",
            next: "init",
            progress: 0,
            elapsed: Duration::new(0, 0),
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
                elapsed: Duration::new(0, 0),
            },
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Success,
                elapsed,
                ..
            } => EmuDynarecPipelineReport {
                elapsed: *elapsed,
                progress,
                current: "cache success",
                next: "restart",
            },
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Fail(_),
                elapsed,
                ..
            } => EmuDynarecPipelineReport {
                elapsed: *elapsed,
                progress,
                current: "cache fail",
                next: "fetch",
            },
            EmuDynarecPipeline::Fetched { elapsed, .. } => EmuDynarecPipelineReport {
                elapsed: *elapsed,
                progress,
                current: "fetched",
                next: "emit",
            },
            EmuDynarecPipeline::Emitted { elapsed, .. } => EmuDynarecPipelineReport {
                elapsed: *elapsed,
                progress,
                current: "emitted",
                next: "call",
            },
            EmuDynarecPipeline::Called { elapsed, .. } => EmuDynarecPipelineReport {
                elapsed: *elapsed,
                progress,
                current: "called",
                next: "cache",
            },
            EmuDynarecPipeline::Cached { elapsed, .. } => EmuDynarecPipelineReport {
                elapsed: *elapsed,
                progress,
                current: "cached",
                next: "restart",
            },
        }
    }
}

pub struct EmuDynarecPipelineIterator<'e, 'j> {
    pipeline: EmuDynarecPipeline,
    emu: &'e mut Emu,
    jit: &'j mut JIT,
}

impl<'e, 'j> Iterator for EmuDynarecPipelineIterator<'e, 'j> {
    type Item = EmuDynarecPipeline;

    fn next(&mut self) -> Option<Self::Item> {
        let pipe = self.pipeline.clone().step(self.emu, self.jit).ok()?;
        self.pipeline = pipe;
        Some(self.pipeline.clone())
    }
}

impl EmuDynarecPipeline {
    #[must_use]
    pub fn iter<'e, 'j>(emu: &'e mut Emu, jit: &'j mut JIT) -> EmuDynarecPipelineIterator<'e, 'j> {
        EmuDynarecPipelineIterator {
            pipeline: EmuDynarecPipeline::Init { pc: emu.cpu.pc },
            emu,
            jit,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use pretty_assertions::{assert_eq, assert_matches};

    use crate::{
        Emu,
        dynarec::{TryCacheSummary, pipeline::EmuDynarecPipeline},
        jit::{JIT, hash_ops},
    };

    #[test]
    fn test_hash() -> color_eyre::Result<()> {
        let mut emu = Emu::default();
        let mut jit = JIT::default();
        emu.load_bios()?;
        emu.jump_to_bios();
        let mut pipeline = EmuDynarecPipeline::from_emu(&emu);
        pipeline = pipeline.step(&mut emu, &mut jit)?;
        let fetched = pipeline.clone();

        assert_matches!(fetched, EmuDynarecPipeline::Fetched { .. });
        let f = fetched.as_fetch().unwrap();
        let hash1a = hash_ops(&f.decoded_ops);
        let hash1b = hash_ops(&f.decoded_ops);
        assert_eq!(hash1a, hash1b, "Hashing same data should be deterministic!");

        pipeline = pipeline.run_once(&mut emu, &mut jit)?;
        emu.jump_to_bios();
        pipeline = pipeline
            .step(&mut emu, &mut jit)?
            .step(&mut emu, &mut jit)?
            .step(&mut emu, &mut jit)?;
        assert_matches!(pipeline, EmuDynarecPipeline::TriedCache { .. });

        assert_matches!(&fetched, pipeline);
        if let (
            EmuDynarecPipeline::Fetched { fetch: fetch_1, .. },
            EmuDynarecPipeline::TriedCache { fetch: fetch_2, .. },
        ) = (&fetched, &pipeline)
        {
            assert_eq!(fetch_1.decoded_ops, fetch_2.decoded_ops);
            assert_eq!(
                hash_ops(&fetch_1.decoded_ops),
                hash_ops(&fetch_2.decoded_ops)
            );
        }
        assert_matches!(
            pipeline,
            EmuDynarecPipeline::TriedCache {
                result: TryCacheSummary::Success,
                ..
            }
        );

        Ok(())
    }
}
