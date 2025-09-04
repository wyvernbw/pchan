use std::fmt::Display;

use tracing::instrument;

use crate::{
    cpu::op::{
        Op, PrimaryOp, SecondaryOp,
        add::{AddImmOp, AddOp},
        jump::{JOp, JalOp, JalrOp, JrOp},
        load::LoadOp,
        store::StoreOp,
        sub::SubOp,
    },
    memory::{Addr, Address, KSEG0Addr, Memory, PhysAddr, ToWord},
};

pub(crate) mod op;
#[cfg(test)]
pub mod pipeline_tests;

/// # Cpu
/// models the PSX cpu.
#[derive(Default, Debug)]
pub struct Cpu {
    reg: [u32; 32],
    pc: u32,
    cycle: usize,
    pipe: PipelineQueue,
    known: usize,
}

/// # PipelineQueue
/// models the PSX cpu pipeline.
/// the pipeline is divided in 5 stages:
/// - fetch: reads an op from memory at `pc` (program counter)
/// - id: decodes the op
/// - ex: executes arithmetic operations and computes the `[rs+imm]` for load ops
/// - mem: reads/writes to memory
/// - wb: writes values to registers
///
/// each stage takes a cycle, however alus implement instruction forwarding effectively making them
/// take 1 cycle instead.
#[derive(Default, Debug)]
struct PipelineQueue {
    fetch_in: Option<FetchOut>,
    fetch_out: Option<FetchOut>,
    id_in: Option<IdIn>,
    id_out: Option<IdOut>,
    ex_in: Option<ExIn>,
    ex_out: Option<ExOut>,
    mem_in: Option<MemIn>,
    mem_out: Option<MemOut>,
    wb_in: Option<WbIn>,
}

impl PipelineQueue {
    #[instrument(skip(mem))]
    pub(crate) fn handle_store(
        mem: &mut Memory,
        rt_value: u32,
        dest: impl Address,
        header: PrimaryOp,
    ) {
        match header {
            PrimaryOp::SB => mem.write(dest, (rt_value as u8).to_word_zeroed()),
            PrimaryOp::SH => mem.write(dest, (rt_value as u16).to_word_zeroed()),
            PrimaryOp::SW => mem.write(dest, rt_value),
            _ => unreachable!(),
        }
    }

    #[instrument(skip(mem))]
    pub(crate) fn handle_load(
        mem: &Memory,
        at: PhysAddr,
        header: PrimaryOp,
        dest: RegisterId,
    ) -> MemOut {
        let value = match header {
            PrimaryOp::LB => mem.read::<u8>(at).to_word_signed(),
            PrimaryOp::LBU => mem.read::<u8>(at).to_word_zeroed(),
            PrimaryOp::LH => mem.read::<u16>(at).to_word_signed(),
            PrimaryOp::LHU => mem.read::<u16>(at).to_word_zeroed(),
            PrimaryOp::LW => mem.read::<u32>(at).to_word_zeroed(),
            _ => unreachable!("invalid header passed to load"),
        };
        MemOut { dest, value }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FetchOut {
    op: Op,
}

type IdIn = FetchOut;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdOut {
    Load(LoadOp),
    Store(StoreOp),
    AluAdd(AddOp),
    AluImmAdd(AddImmOp),
    AluSub(SubOp),
    J(JOp),
    Jal {
        op: JalOp,
        ret: Addr,
        ret_register: RegisterId,
    },
    Jr(Addr),
}

type ExIn = IdOut;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExOut {
    Load {
        header: PrimaryOp,
        rt_value: Addr,
        dest: RegisterId,
    },
    Store {
        header: PrimaryOp,
        rt_value: u32,
        dest: Addr,
    },
    // pass through all the way to WB
    Alu {
        out: u32,
        dest: RegisterId,
    },
    WriteAddress {
        out: Addr,
        dest: RegisterId,
    },
}

type MemIn = ExOut;

#[derive(derive_more::Debug, Clone, Copy, PartialEq, Eq)]
struct MemOut {
    dest: RegisterId,
    #[debug("0x{value:08X}")]
    value: u32,
}

type WbIn = MemOut;
type WbOut = ();

pub enum Interrupt {}

#[derive(Debug, thiserror::Error)]
pub enum Exception {
    /// ## Memory Error
    /// Misalignments
    /// (and probably also KSEG access in User mode)
    #[error("memory exception")]
    MemoryError,
    /// ## Bus Error
    /// Unused Memory Regions (including Gaps in I/O Region)
    /// (unless RAM/BIOS/Expansion mirrors are mapped to "unused" area)
    #[error("bus exception")]
    BusError,
}

impl Cpu {
    #[instrument(skip_all)]
    pub(crate) fn advance_cycle(&mut self) {
        self.cycle += 1;
    }

    /// fetches instruction at the program counter and passes it forward to the
    /// `id` stage. it then advances the program counter.
    #[instrument(skip_all)]
    fn pipeline_fetch(&mut self, mem: &mut Memory) -> Result<(), Exception> {
        let op = mem.read::<Op>(PhysAddr::new(self.pc));
        self.pipe.fetch_in = Some(FetchOut { op });

        if let Some(out) = self.pipe.fetch_out.take() {
            self.pipe.id_in = Some(out);
            tracing::trace!(send = %out.op);
        }

        if let Some(fetch_in) = self.pipe.fetch_in.take() {
            self.pipe.fetch_out = Some(fetch_in);
            tracing::trace!(at = ?self.pc, loaded = %op);
        }

        self.pc += 4;
        Ok(())
    }

    #[instrument(skip_all)]
    fn pipeline_id(&mut self, mem: &mut Memory) -> Result<(), Exception> {
        // flush `id_out`
        if let Some(id_out) = self.pipe.id_out.take() {
            tracing::trace!(send = ?id_out);
            self.pipe.ex_in = Some(id_out);
        }

        // move `id_in` into `id_out`
        if let Some(id_in) = self.pipe.id_in.take() {
            tracing::trace!(recv = %id_in.op);
            // handle NOP
            if id_in.op == Op::NOP {
                return Ok(());
            }
            match id_in.op.primary() {
                PrimaryOp::LW | PrimaryOp::LB | PrimaryOp::LBU | PrimaryOp::LH | PrimaryOp::LHU => {
                    let args = id_in.op.load_op();
                    self.pipe.id_out = Some(IdOut::Load(args));
                }
                PrimaryOp::SB | PrimaryOp::SH | PrimaryOp::SW => {
                    let args: StoreOp = id_in.op.into();
                    self.pipe.id_out = Some(IdOut::Store(args));
                }
                PrimaryOp::ADDI | PrimaryOp::ADDIU => {
                    let args: AddImmOp = id_in.op.into();
                    self.pipe.id_out = Some(IdOut::AluImmAdd(args));
                }
                PrimaryOp::J => {
                    let jop = id_in.op.into();
                    tracing::trace!(?jop);
                    self.pipe.id_out = Some(IdOut::J(jop));
                }
                PrimaryOp::JAL => {
                    self.pipe.id_out = Some(IdOut::Jal {
                        op: JalOp::from(id_in.op),
                        ret: KSEG0Addr::from_phys(self.pc).into(),
                        ret_register: RA,
                    });
                }
                PrimaryOp::SPECIAL => match id_in.op.secondary() {
                    SecondaryOp::ADD | SecondaryOp::ADDU => {
                        let args: AddOp = id_in.op.into();
                        self.pipe.id_out = Some(IdOut::AluAdd(args));
                    }
                    SecondaryOp::SUB | SecondaryOp::SUBU => {
                        let args: SubOp = id_in.op.into();
                        self.pipe.id_out = Some(IdOut::AluSub(args));
                    }
                    SecondaryOp::JR => {
                        let args: JrOp = id_in.op.into();
                        let args = args.to_jump(self);
                        self.pipe.id_out = Some(IdOut::Jr(args.dest));
                    }
                    SecondaryOp::JALR => {
                        // convert jalr to equivalent jal
                        let jalr: JalrOp = id_in.op.into();
                        let ret = KSEG0Addr::from_phys(self.pc).into();
                        tracing::debug!(jalr_dest = self.reg(jalr.dest));
                        let jal = JalOp {
                            dest: Addr(self.reg(jalr.dest)),
                        };
                        self.pipe.id_out = Some(IdOut::Jal {
                            op: jal,
                            ret,
                            ret_register: jalr.ret,
                        })
                    }
                    other => todo!("{other:x?} not yet implemented"),
                },
                other => todo!("{other:x?} not yet implemented"),
            }
        };
        Ok(())
    }

    #[instrument(skip_all)]
    fn pipeline_ex(&mut self, mem: &mut Memory) -> Result<(), Exception> {
        // flush `ex_out`
        if let Some(ex_out) = self.pipe.ex_out.take() {
            tracing::trace!(send = ?ex_out);
            self.pipe.mem_in = Some(ex_out);
        }

        // move `ex_in` into `ex_out`
        if let Some(ex_in) = self.pipe.ex_in.take() {
            tracing::trace!(recv = ?ex_in);
            let ex_out = match ex_in {
                IdOut::Load(load_args) => Some(ExOut::Load {
                    header: load_args.header,
                    rt_value: Addr(
                        self.reg(load_args.rs)
                            .wrapping_add_signed(load_args.imm as i32),
                    ),
                    dest: load_args.rt,
                }),
                IdOut::Store(store) => Some(ExOut::Store {
                    header: store.header,
                    rt_value: self.reg(store.rt),
                    dest: Addr(self.reg(store.rs).wrapping_add_signed(store.imm as i32)),
                }),
                // DONE: implement ALU args
                IdOut::AluAdd(add) => {
                    let out = self.reg(add.rs) + self.reg(add.rt);
                    Some(ExOut::Alu { out, dest: add.rd })
                }
                IdOut::AluImmAdd(add) => {
                    let out = self.reg(add.rs).wrapping_add_signed(add.imm as i32);
                    Some(ExOut::Alu { out, dest: add.rt })
                }
                IdOut::AluSub(sub) => {
                    let out = self.reg(sub.rs) - self.reg(sub.rt);
                    Some(ExOut::Alu { out, dest: sub.rd })
                }
                IdOut::J(jop) => {
                    self.pipe.id_in = None;
                    self.pipe.fetch_out = None;
                    self.control_jump_in_region(jop.dest)?;
                    None
                }
                IdOut::Jal {
                    op,
                    ret,
                    ret_register,
                } => {
                    self.pipe.id_in = None;
                    self.pipe.fetch_out = None;
                    self.control_jump_in_region(op.dest)?;
                    Some(ExOut::WriteAddress {
                        out: ret,
                        dest: ret_register,
                    })
                }
                IdOut::Jr(dest) => {
                    self.pipe.id_in = None;
                    self.pipe.fetch_out = None;
                    self.control_jump_absolute(dest)?;
                    None
                }
            };
            self.pipe.ex_out = ex_out;
        };
        Ok(())
    }

    /// reads value from register, forwarding values if possible
    fn reg(&self, id: RegisterId) -> u32 {
        self.forward_register(id).unwrap_or_else(|| self.reg[id])
    }

    fn forward_register(&self, src: RegisterId) -> Option<u32> {
        // poke ex
        match self.pipe.ex_out {
            Some(ExOut::Alu { out, dest }) if dest == src => {
                return Some(out);
            }
            _ => {}
        }
        // poke mem
        match self.pipe.mem_in {
            Some(MemIn::Alu { out, dest }) if dest == src => {
                return Some(out);
            }
            _ => {}
        };
        match self.pipe.mem_out {
            Some(MemOut { dest, value }) if dest == src => {
                return Some(value);
            }
            _ => {}
        };

        // poke wb
        match self.pipe.wb_in {
            Some(WbIn { dest, value }) if dest == src => {
                return Some(value);
            }
            _ => {}
        };

        None
    }

    #[instrument(skip_all)]
    fn pipeline_mem(&mut self, mem: &mut Memory) -> Result<(), Exception> {
        // flush `mem_out`
        if let Some(mem_out) = self.pipe.mem_out.take() {
            tracing::trace!(send = ?mem_out);
            self.pipe.wb_in = Some(mem_out);
        }

        // move `mem_in` into `mem_out`
        let Some(mem_in) = self.pipe.mem_in.take() else {
            return Ok(());
        };
        tracing::trace!(recv = ?mem_in);
        self.pipe.mem_out = match mem_in {
            MemIn::Load {
                header,
                rt_value: at,
                dest,
            } => {
                // DONE: handle differnet load types
                let at = at.try_into().map_err(|_| Exception::MemoryError)?;
                let read = Some(PipelineQueue::handle_load(mem, at, header, dest));
                tracing::trace!(?read);
                read
            }

            MemIn::Store {
                rt_value,
                dest,
                header,
            } => {
                PipelineQueue::handle_store(mem, rt_value, dest, header);
                None
            }

            MemIn::Alu { out, dest } => {
                // immediate pass through
                self.pipe.wb_in = Some(MemOut { dest, value: out });
                None
            }

            MemIn::WriteAddress { out, dest } => {
                // immediate pass through
                self.pipe.wb_in = Some(MemOut { dest, value: out.0 });
                None
            }
        };
        Ok(())
    }

    #[instrument(err, skip(self))]
    fn control_jump_in_region(
        &mut self,
        target: impl Into<Addr> + core::fmt::Debug,
    ) -> Result<(), Exception> {
        let target: Addr = target.into();
        self.pc = (self.pc & 0xF0000000) + (target.0);
        Ok(())
    }

    #[instrument(err, skip(self))]
    fn control_jump_absolute(
        &mut self,
        target: impl Into<Addr> + core::fmt::Debug,
    ) -> Result<(), Exception> {
        let target: Addr = target.into();
        let target: PhysAddr = target.try_into().map_err(|_| Exception::MemoryError)?;
        self.pc = target.0;
        Ok(())
    }

    #[instrument(skip_all)]
    fn pipeline_wb(&mut self, mem: &mut Memory) -> Result<(), Exception> {
        if let Some(wb_in) = self.pipe.wb_in.take() {
            tracing::trace!(recv = ?wb_in);
            match wb_in.dest {
                dest @ 0..=31 => {
                    self.reg_write(wb_in.dest, wb_in.value);
                }
                register => {
                    tracing::error!("invalid register {register}");
                }
            }
        };
        Ok(())
    }

    fn reg_write(&mut self, dest: RegisterId, value: u32) {
        self.reg[dest] = value;
    }

    #[instrument(fields(cycle = self.cycle, pc = self.pc), skip(mem, self))]
    pub(crate) fn run_cycle(&mut self, mem: &mut Memory) -> Option<Interrupt> {
        // DONE: the rest of the owl

        // tracing::trace!(?self.cycle, "pre cycle: {:#?}", self.pipe);

        let exception: Result<_, Exception> = try {
            self.pipeline_fetch(mem)?;
            self.pipeline_id(mem)?;
            self.pipeline_ex(mem)?;
            self.pipeline_mem(mem)?;
            self.pipeline_wb(mem)?;
        };
        match exception {
            Ok(()) => {}
            Err(exception) => self.exception_handler(mem, exception),
        };

        let reg = self
            .reg
            .iter()
            .enumerate()
            .filter(|(_, x)| **x != 0)
            .map(|(i, x)| format!("{i}:0x{x:08X}"))
            .intersperse(" ".to_string())
            .collect::<String>();
        tracing::trace!(?reg);
        println!();

        None
    }
    pub(crate) fn exception_handler(&mut self, mem: &mut Memory, exception: Exception) {
        tracing::error!("encountered exception: {}", &exception);
    }
}

type RegisterId = usize;

/// zero  Constant (always 0)
const R0: RegisterId = 0;
/// R29 (SP) - Full Decrementing Wasted Stack Pointer
const SP: RegisterId = 29;
/// R31 (RA) Return address (used so by JAL,BLTZAL,BGEZAL opcodes)
const RA: RegisterId = 31;
const PC: RegisterId = 32;

pub(crate) struct Program<T: AsRef<[Op]>>(T);

impl<T: AsRef<[Op]>> Program<T> {
    const fn new(prog: T) -> Self {
        Program(prog)
    }
}

impl<T: AsRef<[Op]>> Display for Program<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0
            .as_ref()
            .iter()
            .map(|op| write!(f, "\n{op}"))
            .try_fold((), |_, el| el)
    }
}
impl<T: AsRef<[Op]>> IntoIterator for Program<T> {
    type Item = Op;

    type IntoIter = impl Iterator<Item = Op>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.as_ref().to_vec().into_iter()
    }
}
