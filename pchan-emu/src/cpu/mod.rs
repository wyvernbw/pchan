use std::{fmt::Display, ops::Range};

use pchan_macros::{OpCode, opcode};
use tracing::instrument;

use crate::memory::{MapAddress, MemRead, MemWrite, Memory, PhysAddr, ToWord};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FetchOut {
    op: Op,
}

type IdIn = FetchOut;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdOut {
    Load(LoadArgs),
    Alu,
}

type ExIn = IdOut;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExOut {
    Load {
        rt_value: PhysAddr,
        dest: RegisterId,
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
                    let args = id_in.op.load_args();
                    self.pipe.id_out = Some(IdOut::Load(args));
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
            self.pipe.ex_out = Some(match ex_in {
                IdOut::Load(load_args) => ExOut::Load {
                    rt_value: PhysAddr::new(
                        self.reg(load_args.rs)
                            .wrapping_add_signed(load_args.imm as i32),
                    ),
                    dest: load_args.rt,
                },
                // TODO: implement ALU args
                IdOut::Alu => ExOut::Alu {
                    out: todo!(),
                    dest: todo!(),
                },
            })
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
        if let Some(mem_in) = self.pipe.mem_in.take() {
            tracing::trace!(recv = ?mem_in);
            let out = match mem_in {
                MemIn::Load {
                    rt_value: value,
                    dest,
                } => {
                    let value = mem.read(value);
                    MemOut { dest, value }
                }
                MemIn::Alu { out, dest } => MemOut { dest, value: out },
            };
            self.pipe.mem_out = Some(out);
        }
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
    fn load_signed<T: MemRead + ToWord>(&mut self, mem: &Memory, args: LoadArgs) {
        let LoadArgs {
            rs: src,
            rt: dest,
            imm: offset,
        } = args;
        let addr = self.reg(src).wrapping_add_signed(offset as i32);
        let data = mem.read::<T>(addr.map());
        self.reg[dest] = data.to_word_signed();
    }

    #[inline]
    fn load_zeroed<T: MemRead + ToWord>(&mut self, mem: &Memory, args: LoadArgs) {
        let LoadArgs {
            rs: src,
            rt: dest,
            imm: offset,
        } = args;
        let addr = self.reg(src).wrapping_add_signed(offset as i32);
        let data = mem.read::<T>(addr.map());
        self.reg[dest] = data.to_word_zeroed();
    }
}

type RegisterId = usize;

const SP: RegisterId = 29;
const RA: RegisterId = 31;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Op(u32);

impl core::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Op")
            .field_with(|f| write!(f, "0x{:08X}", &self.0))
            .finish()
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let primary = self.primary();
        if self.0 == 0 {
            return write!(f, "NOP");
        }
        match primary {
            PrimaryOp::LW | PrimaryOp::LB | PrimaryOp::LBU | PrimaryOp::LH | PrimaryOp::LHU => {
                let args = self.load_args();
                write!(f, "{:?} {} {} {}", primary, args.rs, args.rt, args.imm)
            }
            _ => write!(f, "0x{:08X}", self.0),
        }
    }
}

impl MemRead for Op {
    fn from_slice(buf: &[u8]) -> Result<Self, crate::memory::DerefError> {
        u32::from_slice(buf).map(Op)
    }
}

impl MemWrite for Op {
    fn to_bytes(&self) -> [u8; size_of::<Self>()] {
        self.0.to_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoadArgs {
    rs: RegisterId,
    rt: RegisterId,
    imm: i16,
}

impl Op {
    const NOP: Op = Op(0x00000000);

    #[inline]
    const fn primary(&self) -> PrimaryOp {
        let code = self.0 >> 26;
        PrimaryOp::MAP[code as usize]
    }
    #[inline]
    const fn secondary(&self) -> SecondaryOp {
        let code = self.0 & 0x3F;
        SecondaryOp::MAP[code as usize]
    }
    #[inline]
    const fn bits(&self, range: Range<u8>) -> u32 {
        let mask = (0xFFFFFFFFu32.unbounded_shl(range.start as u32))
            ^ ((0xFFFFFFFFu32).unbounded_shl(range.end as u32));
        (self.0 & mask).unbounded_shr(range.start as u32)
    }

    const fn load_args(&self) -> LoadArgs {
        LoadArgs {
            rs: self.bits(21..26) as usize,
            rt: self.bits(16..21) as usize,
            imm: self.bits(0..16) as i16,
        }
    }
}

#[repr(u8)]
#[derive(OpCode, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryOp {
    SPECIAL = 0x00,
    BCONDZ = 0x01,
    J = 0x02,
    JAL = 0x03,
    BEQ = 0x04,
    BNE = 0x05,
    BLEZ = 0x06,
    BGTZ = 0x07,

    ADDI = 0x08,
    ADDIU = 0x09,
    SLTI = 0x0A,
    SLTIU = 0x0B,
    ANDI = 0x0C,
    ORI = 0x0D,
    XORI = 0x0E,
    LUI = 0x0F,

    COP0 = 0x10,
    COP1 = 0x11,
    COP2 = 0x12,
    COP3 = 0x13,

    LB = 0x20,
    LH = 0x21,
    LWL = 0x22,
    LW = 0x23,
    LBU = 0x24,
    LHU = 0x25,
    LWR = 0x26,

    SB = 0x28,
    SH = 0x29,
    SWL = 0x2A,
    SW = 0x2B,
    SWR = 0x2E,

    LWC0 = 0x30,
    LWC1 = 0x31,
    LWC2 = 0x32,
    LWC3 = 0x33,

    SWC0 = 0x38,
    SWC1 = 0x39,
    SWC2 = 0x3A,
    SWC3 = 0x3B,

    #[opcode(default)]
    ILLEGAL,
}

#[repr(u8)]
#[derive(OpCode, Debug, Clone, Copy, PartialEq, Eq)]
enum SecondaryOp {
    // Shift instructions
    SLL = 0x00,
    SRL = 0x02,
    SRA = 0x03,
    SLLV = 0x04,
    SRLV = 0x06,
    SRAV = 0x07,

    // Jump instructions
    JR = 0x08,
    JALR = 0x09,

    // Move from/to special registers
    MFHI = 0x10,
    MTHI = 0x11,
    MFLO = 0x12,
    MTLO = 0x13,

    // Multiply/Divide
    MULT = 0x18,
    MULTU = 0x19,
    DIV = 0x1A,
    DIVU = 0x1B,

    // Arithmetic
    ADD = 0x20,
    ADDU = 0x21,
    SUB = 0x22,
    SUBU = 0x23,
    AND = 0x24,
    OR = 0x25,
    XOR = 0x26,
    NOR = 0x27,
    SLT = 0x2A,
    SLTU = 0x2B,

    // System
    SYSCALL = 0x0C,
    BREAK = 0x0D,

    #[opcode(default)]
    ILLEGAL,
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn test_primary_op_decoding() {
        // Example: ADDI instruction (opcode 0x08)
        let instr = Op(0x20010001); // 0010 0000 0000 0001 0000 0000 0000 0001
        assert_eq!(instr.primary(), PrimaryOp::ADDI);

        // Example: J instruction (opcode 0x02)
        let instr = Op(0x08000000); // 0000 1000 0000 0000 0000 0000 0000 0000
        assert_eq!(instr.primary(), PrimaryOp::J);

        // Example: ILLEGAL opcode
        let instr = Op(0xFC000000); // 1111 1100 ...
        assert_eq!(instr.primary(), PrimaryOp::ILLEGAL);
    }

    #[test]
    fn test_secondary_op_decoding() {
        // Example: ADD instruction (secondary 0x20)
        let instr = Op(0x00000020); // SPECIAL primary, ADD secondary
        assert_eq!(instr.secondary(), SecondaryOp::ADD);

        // Example: JR instruction (secondary 0x08)
        let instr = Op(0x00000008);
        assert_eq!(instr.secondary(), SecondaryOp::JR);

        // Example: ILLEGAL secondary opcode
        let instr = Op(0x0000003F); // 0x3F = 63
        assert_eq!(instr.secondary(), SecondaryOp::ILLEGAL);
    }

    #[test]
    fn test_register_access() {
        let mut cpu = Cpu {
            reg: [0; 32],
            ..Default::default()
        };
        cpu.reg[SP] = 0xDEADBEEF;
        assert_eq!(cpu.reg[SP], 0xDEADBEEF);

        cpu.reg[1] = 42;
        assert_eq!(cpu.reg[1], 42);
    }

    #[test]
    fn test_bits_basic() {
        let op = Op(0b1111_0000_1010_1100_0101_0011_1001_0110);

        // Extract lower 4 bits (0..4)
        assert_eq!(op.bits(0..4), 0b0110);

        // Extract bits 4..8
        assert_eq!(op.bits(4..8), 0b1001);

        // Extract bits 8..16
        assert_eq!(op.bits(8..16), 0b0101_0011);

        // Extract bits 16..24
        assert_eq!(op.bits(16..24), 0b1010_1100);

        // Extract bits 24..32
        assert_eq!(op.bits(24..32), 0b1111_0000);
    }

    #[test]
    fn test_bits_single_bit() {
        let op = Op(0b1010_1010);

        // Each bit individually
        for i in 0..8 {
            let expected = (op.0 >> i) & 1;
            assert_eq!(op.bits(i..i + 1), expected);
        }
    }

    #[test]
    fn test_bits_full_width() {
        let op = Op(0xDEAD_BEEF);
        assert_eq!(op.bits(0..32), 0xDEAD_BEEF);
    }

    #[test]
    fn test_bits_top_edge() {
        let op = Op(0x8000_0001);

        // top bit only
        assert_eq!(op.bits(31..32), 0b1);

        // bottom bit only
        assert_eq!(op.bits(0..1), 0b1);
    }
}

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
