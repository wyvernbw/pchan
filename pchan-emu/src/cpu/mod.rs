use pchan_macros::{OpCode, opcode};

pub struct Cpu {
    reg: [u32; 32],
}

type RegisterId = usize;

const SP: RegisterId = 29;

pub struct Op(u32);

impl Op {
    #[inline]
    fn primary(&self) -> PrimaryOp {
        let code = self.0 >> 26;
        PrimaryOp::MAP[code as usize]
    }
    #[inline]
    fn secondary(&self) -> SecondaryOp {
        let code = self.0 & 0x3F;
        SecondaryOp::MAP[code as usize]
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
        let mut cpu = Cpu { reg: [0; 32] };
        cpu.reg[SP] = 0xDEADBEEF;
        assert_eq!(cpu.reg[SP], 0xDEADBEEF);

        cpu.reg[1] = 42;
        assert_eq!(cpu.reg[1], 42);
    }
}
