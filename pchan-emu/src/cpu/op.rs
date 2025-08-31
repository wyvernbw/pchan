pub(crate) mod load;
pub(crate) mod store;

use std::{fmt::Display, ops::Range, u8};

use pchan_macros::OpCode;

use crate::{
    cpu::op::store::StoreOp,
    memory::{MemRead, MemWrite},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Op(pub(crate) u32);

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
                let args = self.load_op();
                write!(f, "{:?} {} {} {}", primary, args.rt, args.rs, args.imm)
            }
            PrimaryOp::SB | PrimaryOp::SH | PrimaryOp::SW => {
                let args: StoreOp = (*self).into();
                write!(f, "{}", args)
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

impl MemWrite<{ size_of::<Op>() }> for &Op {
    fn to_bytes(&self) -> [u8; size_of::<Op>()] {
        self.0.to_bytes()
    }
}

impl Op {
    pub(crate) const NOP: Op = Op(0x00000000);

    #[inline]
    pub(crate) const fn primary(&self) -> PrimaryOp {
        let code = self.0 >> 26;
        PrimaryOp::MAP[code as usize]
    }
    #[inline]
    pub(crate) const fn secondary(&self) -> SecondaryOp {
        let code = self.0 & 0x3F;
        SecondaryOp::MAP[code as usize]
    }
    #[inline]
    pub(crate) const fn bits(&self, range: Range<u8>) -> u32 {
        let mask = (0xFFFFFFFFu32.unbounded_shl(range.start as u32))
            ^ ((0xFFFFFFFFu32).unbounded_shl(range.end as u32));
        (self.0 & mask).unbounded_shr(range.start as u32)
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
pub(crate) enum SecondaryOp {
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
    use crate::cpu::op::SecondaryOp;

    use super::*;
    use pchan_utils::setup_tracing;
    use pretty_assertions::{assert_eq, assert_ne};
    use rstest::rstest;

    #[rstest]
    fn test_primary_op_decoding(setup_tracing: ()) {
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

    #[rstest]
    fn test_secondary_op_decoding(setup_tracing: ()) {
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

    #[rstest]
    fn test_bits_basic(setup_tracing: ()) {
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

    #[rstest]
    fn test_bits_single_bit(setup_tracing: ()) {
        let op = Op(0b1010_1010);

        // Each bit individually
        for i in 0..8 {
            let expected = (op.0 >> i) & 1;
            assert_eq!(op.bits(i..i + 1), expected);
        }
    }

    #[rstest]
    fn test_bits_full_width(setup_tracing: ()) {
        let op = Op(0xDEAD_BEEF);
        assert_eq!(op.bits(0..32), 0xDEAD_BEEF);
    }

    #[rstest]
    fn test_bits_top_edge(setup_tracing: ()) {
        let op = Op(0x8000_0001);

        // top bit only
        assert_eq!(op.bits(31..32), 0b1);

        // bottom bit only
        assert_eq!(op.bits(0..1), 0b1);
    }
}
