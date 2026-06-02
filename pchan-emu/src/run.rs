use std::sync::Arc;
use std::time::{Duration, Instant};

use kanal::Sender;
use pchan_bind::ringbuf::storage::Heap;
use pchan_bind::ringbuf::traits::{Consumer, Producer, Split};
use pchan_bind::ringbuf::wrap::caching::Caching;
use pchan_bind::ringbuf::{HeapRb, SharedRb};
use pchan_utils::{Chan, hex};

use crate::Emu;
use crate::cpu::interp::{Interpreter, InterpreterResult};
use crate::dynarec_v2::emitters::{DecodedOp, DynarecOp, EmitCtx};
use crate::dynarec_v2::{Dynarec, DynarecBlock, PipelineCompileError, run_step};
use crate::memory::mb;

#[derive(derive_more::Debug)]
pub struct Runner {
    interpreter:     Interpreter,
    pub(crate) mode: RunnerMode,
    pub config:      RunnerConfig,
    transport:       Transport,
    #[debug(skip)]
    actor_tx:        Caching<Arc<SharedRb<Heap<CompileActorMsg>>>, true, false>,
    own_dynarec:     Dynarec,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum RunnerMode {
    #[default]
    Dynarec,
    Interpreter,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct RunnerConfig {
    pub force_mode: Option<RunnerMode>,
}

#[derive(Debug)]
enum CompileActorMsg {
    Op(u32, DecodedOp),
    Kill,
}

enum CompileActorResponse {
    Compiled(Result<DynarecBlock, PipelineCompileError>),
}

#[derive(Debug, Clone)]
struct Transport {
    out_chan: Chan<CompileActorResponse>,
}

impl Transport {
    fn new() -> Self {
        Transport {
            out_chan: kanal::bounded(1024),
        }
    }
}

impl Default for Transport {
    fn default() -> Self {
        Transport::new()
    }
}

impl Runner {
    pub fn new() -> Self {
        let transport = Transport::new();
        let rb = HeapRb::<CompileActorMsg>::new(mb(32));
        let (prod, cons) = rb.split();
        let transport_2 = transport.clone();
        std::thread::spawn(move || {
            Self::compile_actor(cons, transport_2.out_chan.0);
        });
        Self {
            interpreter: Interpreter::default(),
            mode: RunnerMode::default(),
            config: RunnerConfig::default(),
            transport,
            actor_tx: prod,
            own_dynarec: Dynarec::default(),
        }
    }

    pub fn with_config(mut self, config: RunnerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn execute(&mut self, emu: &mut Emu) {
        match self.config.force_mode {
            Some(RunnerMode::Interpreter) => {
                let (result, _, _) = self.interpreter.run_instruction(emu);
                match result {
                    InterpreterResult::None => {}
                    _ => return,
                }
            }
            Some(RunnerMode::Dynarec) => {
                run_step(emu, &mut self.own_dynarec);
            }
            None => loop {
                if let Ok(Some(CompileActorResponse::Compiled(Ok(block)))) =
                    self.transport.out_chan.1.try_recv()
                {
                    tracing::info!("got compiled block: {}", hex(block.pc));
                    emu.dynarec_cache.insert(block.pc, block);
                }

                match self.mode {
                    RunnerMode::Dynarec => {
                        let pc = emu.cpu.pc;
                        let block = match emu.dynarec_cache.remove(pc) {
                            None => {
                                self.mode = RunnerMode::Interpreter;
                                continue;
                            }
                            Some(block) => block,
                        };
                        block(emu, false);
                        emu.dynarec_cache.insert(pc, block);
                        return;
                    }
                    RunnerMode::Interpreter => {
                        let (result, pc, op) = self.interpreter.run_instruction(emu);
                        if let Some(block) = emu.dynarec_cache.remove(pc) {
                            self.mode = RunnerMode::Dynarec;
                            block(emu, false);
                            emu.dynarec_cache.insert(pc, block);
                            continue;
                        };
                        self.actor_tx.try_push(CompileActorMsg::Op(pc, op)).unwrap();
                        match result {
                            InterpreterResult::None => {}
                            _ => {
                                return;
                            }
                        }
                    }
                }
            },
        }
    }

    fn compile_actor(
        mut rx: Caching<Arc<SharedRb<Heap<CompileActorMsg>>>, false, true>,
        tx: Sender<CompileActorResponse>,
    ) {
        enum ActorState {
            Idle,
            Compiling(CompileState),
            DelaySlot(CompileState),
        }
        struct CompileState {
            init_pc:        u32,
            pc:             u32,
            pc_updated:     bool,
            cycles:         u32,
            scratch_cursor: u8,
            op_count:       usize,
        }

        impl CompileState {
            pub fn new(pc: u32) -> Self {
                Self {
                    init_pc: pc,
                    pc,
                    pc_updated: false,
                    cycles: 0,
                    scratch_cursor: 0,
                    op_count: 0,
                }
            }
            pub fn emit_op(&mut self, dynarec: &mut Dynarec, op: DecodedOp) {
                self.pc_updated |= op
                    .emit(EmitCtx {
                        dynarec,
                        pc: self.pc,
                        d_clock: self.cycles,
                        delay_slot: false,
                        scratch_cursor: &mut self.scratch_cursor,
                    })
                    .pc_updated;
                if let Some(pre_scheduled) = dynarec.pop_scheduled_at(self.pc) {
                    // let cache_boundary_check = (pre_scheduled.pc.saturating_sub(0x4)) >> 2;
                    // let boundary = cache_boundary_check != 0
                    //     && cache_boundary_check.is_multiple_of(PAGE_LEN as u32);
                    // assert!(!boundary, "delay slot is on cache boundary");
                    self.pc_updated |= pre_scheduled
                        .emitter
                        .call((EmitCtx {
                            dynarec,
                            pc: pre_scheduled.pc,
                            d_clock: self.cycles,
                            delay_slot: true,
                            scratch_cursor: &mut self.scratch_cursor,
                        },))
                        .pc_updated;
                }
                self.cycles += op.cycles() as u32;
                if op.hazard() != 0 {
                    self.cycles -= 1;
                }
                self.op_count += 1;
            }
            pub fn advance(mut self, dynarec: &mut Dynarec, op: DecodedOp) -> ActorState {
                self.emit_op(dynarec, op);
                self = CompileState {
                    pc: self.pc + 4,
                    ..self
                };
                match op.is_boundary() {
                    true => ActorState::DelaySlot(self),
                    false => ActorState::Compiling(self),
                }
            }
        }

        let mut state = ActorState::Idle;
        let mut dynarec = Dynarec::default();
        let sleep_duration = Duration::from_millis(2);
        let mut last_packet = Instant::now();
        loop {
            let msg = match rx.try_pop() {
                Some(msg) => msg,
                None => {
                    if last_packet.elapsed() > Duration::from_millis(3) {
                        std::thread::sleep(sleep_duration);
                    }
                    std::thread::yield_now();
                    continue;
                }
            };
            last_packet = Instant::now();
            match msg {
                CompileActorMsg::Op(pc, op) => {
                    match state {
                        ActorState::Idle => {
                            dynarec.reset();
                            dynarec.emit_block_prelude();
                            let compile_state = CompileState::new(pc);
                            state = compile_state.advance(&mut dynarec, op);
                        }
                        ActorState::Compiling(compile_state) => {
                            state = compile_state.advance(&mut dynarec, op);
                        }
                        ActorState::DelaySlot(mut compile_state) => {
                            compile_state.emit_op(&mut dynarec, op);
                            state = ActorState::Idle;

                            // drain scheduler
                            while let Some(emitter) = dynarec.scheduler.queue.pop() {
                                tracing::trace!("draining {:?}", emitter);
                                compile_state.pc_updated |= emitter
                                    .emitter
                                    .call((EmitCtx {
                                        dynarec:        &mut dynarec,
                                        pc:             emitter.pc,
                                        d_clock:        compile_state.cycles,
                                        // this happens in the delay slot basically
                                        delay_slot:     true,
                                        scratch_cursor: &mut compile_state.scratch_cursor,
                                    },))
                                    .pc_updated;
                            }

                            let new_pc = match compile_state.pc_updated {
                                true => None,
                                false => Some(compile_state.pc + 0x4),
                            };
                            dynarec.emit_block_epilogue(compile_state.cycles, new_pc, true);

                            let func =
                                dynarec
                                    .finalize()
                                    .map_err(|_| PipelineCompileError)
                                    .map(|func| DynarecBlock {
                                        function: func,
                                        op_count: compile_state.op_count as u32,
                                        pc:       compile_state.init_pc,
                                    });
                            tx.send(CompileActorResponse::Compiled(func))
                                .expect("channel closed");
                        }
                    }
                }
                CompileActorMsg::Kill => return,
            }
        }
    }
}

impl Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}
