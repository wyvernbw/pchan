use tracing::instrument;

use crate::{
    cpu::op::{Op, PrimaryOp, load::LoadOp, store::StoreOp},
    memory::{Address, MapAddress, MemRead, Memory, PhysAddr, ToWord},
};

pub(crate) mod op;

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
    Alu,
}

type ExIn = IdOut;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExOut {
    Load {
        rt_value: PhysAddr,
        dest: RegisterId,
    },
    Store {
        header: PrimaryOp,
        rt_value: u32,
        dest: PhysAddr,
    },
    // pass through all the way to WB
    Alu {
        out: u32,
        dest: RegisterId,
    },
}

type MemIn = ExOut;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemOut {
    dest: RegisterId,
    value: u32,
}

type WbIn = MemOut;
type WbOut = ();

pub enum Interrupt {}

impl Cpu {
    #[instrument(skip_all)]
    pub(crate) fn advance_cycle(&mut self) {
        self.cycle += 1;
    }

    /// fetches instruction at the program counter and passes it forward to the
    /// `id` stage. it then advances the program counter.
    #[instrument(skip_all)]
    fn pipeline_fetch(&mut self, mem: &mut Memory) {
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
    }

    #[instrument(skip_all)]
    fn pipeline_id(&mut self, mem: &mut Memory) {
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
                return;
            }
            match id_in.op.primary() {
                PrimaryOp::LW | PrimaryOp::LB | PrimaryOp::LBU | PrimaryOp::LH | PrimaryOp::LHU => {
                    let args = id_in.op.load_op();
                    self.pipe.id_out = Some(IdOut::Load(args));
                }
                PrimaryOp::SB | PrimaryOp::SH | PrimaryOp::SW => {
                    let args: StoreOp = id_in.op.into();
                    // send it straight to ex
                    self.pipe.ex_in = Some(IdOut::Store(args));
                }
                other => todo!("{other:x?} not yet implemented"),
            }
        }
    }

    #[instrument(skip_all)]
    fn pipeline_ex(&mut self, mem: &mut Memory) {
        // flush `ex_out`
        if let Some(ex_out) = self.pipe.ex_out.take() {
            tracing::trace!(send = ?ex_out);
            self.pipe.mem_in = Some(ex_out);
        }

        // move `ex_in` into `ex_out`
        if let Some(ex_in) = self.pipe.ex_in.take() {
            tracing::trace!(recv = ?ex_in);
            let ex_out = match ex_in {
                IdOut::Load(load_args) => ExOut::Load {
                    rt_value: PhysAddr::new(
                        self.reg(load_args.rs)
                            .wrapping_add_signed(load_args.imm as i32),
                    ),
                    dest: load_args.rt,
                },
                IdOut::Store(store) => ExOut::Store {
                    header: store.header,
                    rt_value: self.reg(store.rt),
                    dest: PhysAddr::new(self.reg(store.rs).wrapping_add_signed(store.imm as i32)),
                },
                // TODO: implement ALU args
                IdOut::Alu => ExOut::Alu {
                    out: todo!(),
                    dest: todo!(),
                },
            };
            self.pipe.ex_out = Some(ex_out);
        }
    }

    /// reads value from register, forwarding values if possible
    fn reg(&self, id: RegisterId) -> u32 {
        self.forward_register(id).unwrap_or_else(|| self.reg[id])
    }

    fn forward_register(&self, src: RegisterId) -> Option<u32> {
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
    fn pipeline_mem(&mut self, mem: &mut Memory) {
        // flush `mem_out`
        if let Some(mem_out) = self.pipe.mem_out.take() {
            tracing::trace!(send = ?mem_out);
            self.pipe.wb_in = Some(mem_out);
        }

        // move `mem_in` into `mem_out`
        self.pipe.mem_out = self.pipe.mem_in.take().and_then(|mem_in| {
            tracing::trace!(recv = ?mem_in);
            match mem_in {
                MemIn::Load {
                    rt_value: value,
                    dest,
                } => {
                    // FIXME: handle differnet load types
                    let value = mem.read(value);
                    Some(MemOut { dest, value })
                }

                MemIn::Store {
                    rt_value,
                    dest,
                    header,
                } => {
                    PipelineQueue::handle_store(mem, rt_value, dest, header);
                    None
                }

                MemIn::Alu { out, dest } => Some(MemOut { dest, value: out }),
            }
        });
    }

    #[instrument(skip_all)]
    fn pipeline_wb(&mut self, mem: &mut Memory) {
        if let Some(wb_in) = self.pipe.wb_in.take() {
            tracing::trace!(recv = ?wb_in);
            self.reg_write(wb_in.dest, wb_in.value);
        }
    }

    fn reg_write(&mut self, dest: RegisterId, value: u32) {
        self.reg[dest] = value;
    }

    #[instrument(fields(cycle = self.cycle), skip(mem, self))]
    pub(crate) fn run_cycle(&mut self, mem: &mut Memory) -> Option<Interrupt> {
        // DONE: the rest of the owl

        // tracing::trace!(?self.cycle, "pre cycle: {:#?}", self.pipe);

        self.pipeline_fetch(mem);
        self.pipeline_id(mem);
        self.pipeline_ex(mem);
        self.pipeline_mem(mem);
        self.pipeline_wb(mem);

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

    #[inline]
    fn load_signed<T: MemRead + ToWord>(&mut self, mem: &Memory, args: LoadOp) {
        let LoadOp {
            rs: src,
            rt: dest,
            imm: offset,
            ..
        } = args;
        let addr = self.reg(src).wrapping_add_signed(offset as i32);
        let data = mem.read::<T>(addr.map());
        self.reg[dest] = data.to_word_signed();
    }

    #[inline]
    fn load_zeroed<T: MemRead + ToWord>(&mut self, mem: &Memory, args: LoadOp) {
        let LoadOp {
            rs: src,
            rt: dest,
            imm: offset,
            ..
        } = args;
        let addr = self.reg(src).wrapping_add_signed(offset as i32);
        let data = mem.read::<T>(addr.map());
        self.reg[dest] = data.to_word_zeroed();
    }
}

type RegisterId = usize;

const SP: RegisterId = 29;
const RA: RegisterId = 31;

#[cfg(test)]
pub mod pipeline_tests {
    use super::*;
    use crate::memory::{MemRead, PhysAddr};
    use pchan_utils::setup_tracing;
    use pretty_assertions::{assert_eq, assert_matches};
    use rstest::*;

    #[rstest]
    #[instrument]
    fn test_pipeline_single_load(setup_tracing: ()) {
        tracing::info!("testing single load");
        let mut cpu = Cpu::default();
        let mut mem = Memory::default();

        let instr = 0x8D280004;
        cpu.reg[9] = 0x1000; // r9 = 0x1000
        mem.write::<u32>(PhysAddr::new(0x0), instr); // Instruction at PC=0
        mem.write::<u32>(PhysAddr::new(0x1004), 0x12345678); // Data at 0x1004

        // Run 5 cycles to complete one load instruction
        for _ in 0..5 {
            cpu.run_cycle(&mut mem);
            cpu.advance_cycle();
        }

        tracing::info!("{:08x}", cpu.reg[8]);

        // Verify $t0 (reg 8) contains the loaded value
        assert_eq!(cpu.reg[8], 0x12345678, "LW should load 0x12345678 into $t0");
    }
    #[rstest]
    #[instrument]
    fn test_pipeline_lb(setup_tracing: ()) {
        tracing::info!("testing LB instruction");
        let mut cpu = Cpu::default();
        let mut mem = Memory::default();

        let instr = 0x81080002; // LB r8, 2(r8)
        cpu.reg[8] = 0x1000;
        mem.write::<u32>(PhysAddr::new(0x0), instr);
        mem.write::<u8>(PhysAddr::new(0x1002), 0xAB); // byte to load

        for _ in 0..5 {
            cpu.run_cycle(&mut mem);
            cpu.advance_cycle();
        }

        assert_eq!(cpu.reg[8], 0xAB000000, "LB should load 0xAB into r8");
    }

    #[rstest]
    #[instrument]
    fn test_pipeline_multiple_loads(setup_tracing: ()) {
        tracing::info!("testing multiple loads in a row");
        let mut cpu = Cpu::default();
        let mut mem = Memory::default();

        let instr1 = Op(0x8D280004); // LW r8, 4(r9)
        tracing::info!(%instr1);
        let instr2 = Op(0x81090001); // LB r9, 1(r8)
        tracing::info!(%instr2);
        let instr3 = Op(0x814A0002); // LBU r10, 2(r10)
        tracing::info!(%instr3);

        cpu.reg[9] = 0x1000;
        cpu.reg[10] = 0x2000;

        mem.write_all(
            PhysAddr::new(0),
            [
                Op::lw(8, 9, 4),
                Op::NOP,
                Op::lb(9, 8, 1),
                Op::lbu(10, 10, 2),
            ],
        );

        mem.write(PhysAddr::new(0), instr1);
        mem.write(PhysAddr::new(4), Op::NOP);
        mem.write(PhysAddr::new(8), instr2);
        mem.write(PhysAddr::new(12), instr3);

        mem.write::<u32>(PhysAddr::new(0x1004), 0x1234); // LW source
        mem.write::<u8>(PhysAddr::new(0x1235), 0x7F); // LB source (0x1235 = r8 + 1)
        mem.write::<u8>(PhysAddr::new(0x2002), 0xFF); // LBU source (r10 + 2)

        // Run enough cycles to complete all three loads
        for _ in 0..15 {
            cpu.run_cycle(&mut mem);
            cpu.advance_cycle();
        }

        assert_eq!(cpu.reg[8], 0x1234, "LW should load 0x1234 into r8");
        assert_eq!(cpu.reg[9], 0x7F000000, "LB should load 0x7F into r9");
        assert_eq!(cpu.reg[10], 0xFF000000, "LBU should load 0xFF into r10");
    }
}
