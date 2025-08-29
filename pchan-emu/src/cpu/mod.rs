use std::{arch::aarch64::uint32x4x2_t, ops::Range};

use pchan_macros::{OpCode, opcode};
use tracing::instrument;

use crate::memory::{Address, MemRead, Memory, PhysAddr, ToWord};

#[derive(Default)]
pub struct Cpu {
    reg: [u32; 32],
    pc: u32,
    cycle: usize,
    running_op: Option<RunningOp>,
    pipe: PipelineQueue,
    known: usize,
}

#[derive(Default)]
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

#[derive(Debug, Clone, Copy)]
struct FetchOut {
    op: Op,
}

type IdIn = FetchOut;

#[derive(Debug, Clone, Copy)]
enum IdOut {
    Load(LoadArgs),
    Alu,
}

type ExIn = IdOut;

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
struct MemOut {
    dest: RegisterId,
    value: u32,
}

type WbIn = MemOut;
type WbOut = ();

#[derive(Debug, Clone, Copy)]
struct RunningOp {
    op: Op,
    cycles_left: i32,
}

pub enum Interrupt {}

impl Cpu {
    #[instrument(skip_all)]
    pub(crate) fn advance_cycle(&mut self) {
        self.cycle += 1;
    }

    fn pipeline_fetch(&mut self, mem: &mut Memory) {
        self.pipe.fetch_in = Some(FetchOut {
            op: mem.read::<Op>(self.pc),
        });

        match self.pipe.fetch_out.take() {
            None => {
                self.pipe.fetch_out = self.pipe.fetch_in;
            }
            Some(out) => {
                self.pipe.id_in = Some(out);
            }
        }
    }

    fn pipeline_id(&mut self, mem: &mut Memory) {
        match (self.pipe.id_in.take(), self.pipe.id_out.take()) {
            (Some(id_in), None) => match id_in.op.primary() {
                PrimaryOp::LW | PrimaryOp::LB | PrimaryOp::LBU | PrimaryOp::LH | PrimaryOp::LHU => {
                    let args = id_in.op.load_args();
                    self.pipe.id_out = Some(IdOut::Load(args));
                }
                _ => todo!(),
            },
            (Some(_), Some(id_out)) => {
                self.pipe.ex_in = Some(id_out);
            }
            (None, None) => {}
            (None, Some(_)) => unreachable!(),
        }
    }

    fn pipeline_ex(&mut self, mem: &mut Memory) {
        match (self.pipe.ex_in.take(), self.pipe.ex_out.take()) {
            (Some(ex_in), None) => {
                self.pipe.ex_out = Some(match ex_in {
                    IdOut::Load(load_args) => ExOut::Load {
                        rt_value: PhysAddr::new(
                            self.forward_register(load_args.rs)
                                .unwrap_or(self.reg[load_args.rs])
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
            (Some(_), Some(ex_out)) => {
                self.pipe.mem_in = Some(ex_out);
            }
            (None, None) => {}
            (None, Some(_)) => unreachable!(),
        }
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

    fn pipeline_mem(&mut self, mem: &mut Memory) {
        match (self.pipe.mem_in.take(), self.pipe.mem_out.take()) {
            (Some(mem_in), None) => {
                self.pipe.mem_out = Some(match mem_in {
                    MemIn::Load {
                        rt_value: value,
                        dest,
                    } => {
                        let value = mem.read(value);
                        MemOut { dest, value }
                    }
                    ExOut::Alu { out, dest } => MemOut { dest, value: out },
                })
            }
            (Some(_), Some(mem_out)) => {
                self.pipe.wb_in = Some(mem_out);
            }
            (None, None) => {}
            (None, Some(_)) => unreachable!(),
        }
    }

    fn pipeline_wb(&mut self, mem: &mut Memory) {
        match self.pipe.wb_in.take() {
            Some(wb_in) => {
                self.reg_write(wb_in.dest, wb_in.value);
            }
            None => {}
        }
    }

    fn reg_write(&mut self, dest: RegisterId, value: u32) {
        self.reg[dest] = value;
    }

    #[instrument(skip_all)]
    pub(crate) fn run_cycle(&mut self, mem: &mut Memory) -> Option<Interrupt> {
        // DONE: the rest of the owl

        self.pipeline_fetch(mem);
        self.pipeline_id(mem);
        self.pipeline_ex(mem);
        self.pipeline_mem(mem);
        self.pipeline_wb(mem);

        None
    }

    #[inline]
    fn load_signed<T: MemRead + ToWord>(&mut self, mem: &Memory, args: LoadArgs) {
        let LoadArgs {
            rs: src,
            rt: dest,
            imm: offset,
        } = args;
        let addr = self.reg[src].wrapping_add_signed(offset as i32);
        let data = mem.read::<T>(addr);
        self.reg[dest] = data.to_word_signed();
    }

    #[inline]
    fn load_zeroed<T: MemRead + ToWord>(&mut self, mem: &Memory, args: LoadArgs) {
        let LoadArgs {
            rs: src,
            rt: dest,
            imm: offset,
        } = args;
        let addr = self.reg[src].wrapping_add_signed(offset as i32);
        let data = mem.read::<T>(addr);
        self.reg[dest] = data.to_word_zeroed();
    }
}

type RegisterId = usize;

const SP: RegisterId = 29;
const RA: RegisterId = 31;

#[derive(Debug, Clone, Copy)]
pub struct Op(u32);

impl MemRead for Op {
    fn from_slice(buf: &[u8]) -> Result<Self, crate::memory::DerefError> {
        u32::from_slice(buf).map(Op)
    }
}

#[derive(Debug, Clone, Copy)]
struct LoadArgs {
    rs: RegisterId,
    rt: RegisterId,
    imm: i16,
}

impl Op {
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

    fn load_args(&self) -> LoadArgs {
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
    use pretty_assertions::assert_eq;

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
