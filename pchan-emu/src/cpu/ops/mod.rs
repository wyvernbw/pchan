//! # Encoding
//!
//! Primary opcode field (Bit 26..31)
//!   00h=SPECIAL 08h=ADDI  10h=COP0 18h=N/A   20h=LB   28h=SB   30h=LWC0 38h=SWC0
//!   01h=BcondZ  09h=ADDIU 11h=COP1 19h=N/A   21h=LH   29h=SH   31h=LWC1 39h=SWC1
//!   02h=J       0Ah=SLTI  12h=COP2 1Ah=N/A   22h=LWL  2Ah=SWL  32h=LWC2 3Ah=SWC2
//!   03h=JAL     0Bh=SLTIU 13h=COP3 1Bh=N/A   23h=LW   2Bh=SW   33h=LWC3 3Bh=SWC3
//!   04h=BEQ     0Ch=ANDI  14h=N/A  1Ch=N/A   24h=LBU  2Ch=N/A  34h=N/A  3Ch=N/A
//!   05h=BNE     0Dh=ORI   15h=N/A  1Dh=N/A   25h=LHU  2Dh=N/A  35h=N/A  3Dh=N/A
//!   06h=BLEZ    0Eh=XORI  16h=N/A  1Eh=N/A   26h=LWR  2Eh=SWR  36h=N/A  3Eh=N/A
//!   07h=BGTZ    0Fh=LUI   17h=N/A  1Fh=N/A   27h=N/A  2Fh=N/A  37h=N/A  3Fh=N/A
//!
//! Secondary opcode field (Bit 0..5) (when Primary opcode = 00h)
//!
//!   00h=SLL   08h=JR      10h=MFHI 18h=MULT  20h=ADD  28h=N/A  30h=N/A  38h=N/A
//!   01h=N/A   09h=JALR    11h=MTHI 19h=MULTU 21h=ADDU 29h=N/A  31h=N/A  39h=N/A
//!   02h=SRL   0Ah=N/A     12h=MFLO 1Ah=DIV   22h=SUB  2Ah=SLT  32h=N/A  3Ah=N/A
//!   03h=SRA   0Bh=N/A     13h=MTLO 1Bh=DIVU  23h=SUBU 2Bh=SLTU 33h=N/A  3Bh=N/A
//!   04h=SLLV  0Ch=SYSCALL 14h=N/A  1Ch=N/A   24h=AND  2Ch=N/A  34h=N/A  3Ch=N/A
//!   05h=N/A   0Dh=BREAK   15h=N/A  1Dh=N/A   25h=OR   2Dh=N/A  35h=N/A  3Dh=N/A
//!   06h=SRLV  0Eh=N/A     16h=N/A  1Eh=N/A   26h=XOR  2Eh=N/A  36h=N/A  3Eh=N/A
//!   07h=SRAV  0Fh=N/A     17h=N/A  1Fh=N/A   27h=NOR  2Fh=N/A  37h=N/A  3Fh=N/A
//!
//! [PSX-SPX Cpu Specifications](https://psx-spx.consoledev.net/cpuspecifications/)

use arbitrary_int::*;
use bitbybit::bitfield;
use derive_more as d;
use pchan_macros::Encode;

pub const fn nop() -> OpCode {
    OpCode::NOP_FIELDS
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[display("nop")]
pub struct Nop;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[display("halt")]
pub struct HaltBlock;

#[bitfield(u32)]
#[derive(PartialEq, Eq, Debug)]
#[derive_const(Default)]
pub struct OpCode {
    #[bits(0..=5, rw)]
    pub funct:  u6,
    #[bits(6..=10, rw)]
    pub shamt:  u5,
    #[bits(11..=15, rw)]
    pub rd:     u5,
    #[bits(16..=20, rw)]
    pub rt:     u5,
    #[bits(21..=25, rw)]
    pub rs:     u5,
    #[bits(26..=31, rw)]
    pub opcode: u6,
    #[bits(0..=15, rw)]
    pub imm16:  u16,
    #[bits(0..=25, rw)]
    pub imm26:  u26,
    #[bits(26..=27, rw)]
    pub cop:    u2,
}

impl OpCode {
    pub const NOP_FIELDS: Self = Self::default();
    pub const HALT: Self = OpCode::new_with_raw_value(69420);
}

impl From<u32> for OpCode {
    fn from(value: u32) -> Self {
        OpCode::new_with_raw_value(value)
    }
}

#[cfg(test)]
mod decode_tests;

use pchan_utils::hex;

use crate::cpu::reg_str;

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[display("addiu ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
#[encode(opcode = 0x09)]
pub struct Addiu {
    pub rs:    u8,
    pub rt:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[display("addu ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
#[encode(opcode = 0x00, funct = 0x21)]
pub struct Addu {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x24)]
#[display("and ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct And {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0c)]
#[display("andi ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Andi {
    pub rs:    u8,
    pub rt:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x04)]
#[display("beq ${}, ${}, {}", reg_str(self.rs), reg_str(self.rt), hex(self.imm16 * 4 + 4))]
pub struct Beq {
    pub rs:    u8,
    pub rt:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x1, rt = 0x1)]
#[display("bgez ${}, {}", reg_str(self.rs), hex(self.imm16))]
pub struct Bgez {
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x07)]
#[display("bgtz ${}, {}", reg_str(self.rs), hex(self.imm16))]
pub struct Bgtz {
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x06)]
#[display("blez ${}, {}", reg_str(self.rs), hex(self.imm16))]
pub struct Blez {
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x01, rt = 0x0)]
#[display("bltz ${}, {}", reg_str(self.rs), hex(self.imm16))]
pub struct Bltz {
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x05)]
#[display("bne ${}, ${}, {}", reg_str(self.rs), reg_str(self.rt), hex(self.imm16 * 4 + 4))]
pub struct Bne {
    pub rs:    u8,
    pub rt:    u8,
    pub imm16: i16,
}

#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
#[display("break")]
pub struct Break;

pub const fn brk() -> OpCode {
    OpCode::default()
        .with_funct(u6::new(0x0D))
        .with_opcode(u6::new(0x0))
}

#[derive(Encode, Debug, Clone, Copy, d::Display, Hash, PartialEq, Eq)]
#[encode(opcode = 0x0, funct = 0x1A)]
#[display("div ${}, ${}", reg_str(self.rs), reg_str(self.rt))]
pub struct Div {
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, d::Display, Hash, PartialEq, Eq)]
#[encode(opcode = 0x0, funct = 0x1B)]
#[display("divu ${},${}", reg_str(self.rs), reg_str(self.rt))]
pub struct Divu {
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x02)]
#[display("j {}", hex(self.imm26 * 4))]
pub struct J {
    pub imm26: i32,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x03)]
#[display("jal {}", hex(self.imm26 * 4))]
#[allow(clippy::upper_case_acronyms)]
pub struct Jal {
    pub imm26: i32,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x09)]
#[display("jalr ${}, ${}", reg_str(self.rd), reg_str(self.rs))]
pub struct Jalr {
    pub rd: u8,
    pub rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x08)]
#[display("jr ${}", reg_str(self.rs))]
pub struct Jr {
    pub rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x20)]
#[display("lb ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Lb {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x24)]
#[display("lbu ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Lbu {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x21)]
#[display("lh ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Lh {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x25)]
#[display("lhu ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Lhu {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0f)]
#[display("lui ${}, {}", reg_str(self.rt), hex(self.imm16))]
pub struct Lui {
    pub rt:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x23)]
#[display("lw ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Lw {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x30, cop = self.cop)]
#[display("lwc{} ${}, ${}, {}", self.cop, reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Lwcn {
    pub cop:   u8,
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

pub fn lwc(n: u8) -> impl Fn(u8, u8, i16) -> OpCode {
    move |rt, rs, imm| Lwcn::new(n, rt, rs, imm).into()
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x10, rs = 0x00, funct = 0x00, cop = self.cop)]
#[display("mfc{} ${}, $r{}", self.cop, reg_str(self.rt), self.rd)]
pub struct Mfcn {
    pub cop: u8,
    pub rt:  u8,
    pub rd:  u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0, funct = 0x10)]
#[display("mfhi ${}", reg_str(self.rd))]
pub struct Mfhi {
    pub rd: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x12)]
#[display("mflo ${}", reg_str(self.rd))]
pub struct Mflo {
    pub rd: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x10, rs = 0x04, funct = 0x00, cop = self.cop)]
#[display("mtc{} ${}, $r{}", self.cop, reg_str(self.rt), self.rd)]
pub struct Mtcn {
    pub cop: u8,
    pub rt:  u8,
    pub rd:  u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x11)]
#[display("mthi ${}", reg_str(self.rs))]
pub struct Mthi {
    rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x13)]
#[display("mtlo ${}", reg_str(self.rs))]
pub struct Mtlo {
    rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x18)]
#[display("mult ${}, ${}", reg_str(self.rs), reg_str(self.rt))]
pub struct Mult {
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x19)]
#[display("multu ${}, ${}", reg_str(self.rs), reg_str(self.rt))]
pub struct Multu {
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x27)]
#[display("nor ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct Nor {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x25)]
#[display("or ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct Or {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0D)]
#[display("ori ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Ori {
    pub rs:    u8,
    pub rt:    u8,
    pub imm16: i16,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[display("rfe")]
pub struct Rfe;

pub fn rfe() -> OpCode {
    OpCode::default()
        .with_opcode(u6::new(0x10))
        .with_rs(u5::new(0x10))
        .with_funct(u6::new(0x10))
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x28)]
#[display("sb ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Sb {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x29)]
#[display("sh ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Sh {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x00, order = "rt_rs")]
#[display("sll ${}, ${}, {}", reg_str(self.rd), reg_str(self.rt), hex(self.shamt))]
pub struct Sll {
    pub rd:    u8,
    pub rt:    u8,
    pub shamt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x04, order = "rt_rs")]
#[display("sllv ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rt), reg_str(self.rs))]
pub struct Sllv {
    pub rd: u8,
    pub rt: u8,
    pub rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x2A)]
#[display("slt ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct Slt {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0A)]
#[display("slti ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Slti {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0B)]
#[display("sltiu ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Sltiu {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: u16,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x2B)]
#[display("sltu ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct Sltu {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x03)]
#[display("sra ${}, ${}, {}", reg_str(self.rd), reg_str(self.rt), hex(self.shamt))]
pub struct Sra {
    pub rd:    u8,
    pub rt:    u8,
    pub shamt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x07)]
#[display("srav ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rt), reg_str(self.rs))]
pub struct Srav {
    pub rd: u8,
    pub rt: u8,
    pub rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x02)]
#[display("srl ${}, ${}, {}", reg_str(self.rd), reg_str(self.rt), hex(self.shamt))]
pub struct Srl {
    pub rd:    u8,
    pub rt:    u8,
    pub shamt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x06, order = "rt_rs")]
#[display("srlv ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rt), reg_str(self.rs))]
pub struct Srlv {
    pub rd: u8,
    pub rt: u8,
    pub rs: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x23)]
#[display("subu ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct Subu {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x2B)]
#[display("sw ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Sw {
    pub rt:    u8,
    pub rs:    u8,
    pub imm16: i16,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, derive_more::Display)]
#[display("syscall")]
pub struct Syscall;

impl From<Syscall> for OpCode {
    fn from(_: Syscall) -> Self {
        let mut op = OpCode::default();
        op.set_funct(u6::new(0xc));
        op
    }
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x00, funct = 0x26)]
#[display("xor ${}, ${}, ${}", reg_str(self.rd), reg_str(self.rs), reg_str(self.rt))]
pub struct Xor {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

#[derive(Encode, Debug, Clone, Copy, Hash, PartialEq, Eq, d::Display)]
#[encode(opcode = 0x0E)]
#[display("xori ${}, ${}, {}", reg_str(self.rt), reg_str(self.rs), hex(self.imm16))]
pub struct Xori {
    pub rs:    u8,
    pub rt:    u8,
    pub imm16: i16,
}
