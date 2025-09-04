use std::fmt::Display;

use crate::{
    cpu::{
        Cpu, RegisterId,
        op::{Op, PrimaryOp, SecondaryOp},
    },
    memory::Addr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JOp {
    pub(crate) dest: Addr,
}

impl const From<Op> for JOp {
    #[inline]
    fn from(value: Op) -> Self {
        JOp {
            dest: Addr(value.bits(0..26) << 2),
        }
    }
}

impl const From<JOp> for Op {
    #[inline]
    fn from(JOp { dest }: JOp) -> Self {
        let imm26 = dest.0 >> 2;
        let imm26 = imm26 & 0x03FF_FFFF; // 26-bit mask
        let i = ((PrimaryOp::J as u32) << 26) | (imm26);
        Self(i)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn j(dest: impl [const] Into<Addr>) -> Op {
        let addr = dest.into();
        JOp { dest: addr }.into()
    }
}

impl Display for JOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "J 0x{:08X}", self.dest.0)
    }
}

/// JAL instruction
/// call dest | jal dest | pc=(pc and F0000000h)+(imm26bit*4),ra=$+8
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JalOp {
    pub(crate) dest: Addr,
}

impl const From<Op> for JalOp {
    #[inline]
    fn from(value: Op) -> Self {
        JalOp {
            dest: Addr(value.bits(0..26) << 2),
        }
    }
}

impl const From<JalOp> for Op {
    #[inline]
    fn from(JalOp { dest }: JalOp) -> Self {
        let imm26 = dest.0 >> 2;
        let imm26 = imm26 & 0x03FF_FFFF; // 26-bit mask
        let i = ((PrimaryOp::JAL as u32) << 26) | (imm26);
        Self(i)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn jal(dest: impl [const] Into<Addr>) -> Op {
        let addr = dest.into();
        JalOp { dest: addr }.into()
    }
}

impl Display for JalOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JAL 0x{:08X}", self.dest.0)
    }
}

/// # JrOp
/// jmp rs | jr rs | pc=rs
pub(crate) struct JrOp {
    reg: RegisterId,
}

impl const From<Op> for JrOp {
    #[inline]
    fn from(value: Op) -> Self {
        JrOp {
            reg: value.bits(21..26) as usize,
        }
    }
}

impl const From<JrOp> for Op {
    #[inline]
    fn from(value: JrOp) -> Self {
        Op(0)
            .with_primary(PrimaryOp::SPECIAL)
            .with_secondary(SecondaryOp::JR)
            .set_bits(21..26, value.reg as u32)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn jr(reg: RegisterId) -> Op {
        JrOp { reg }.into()
    }
}

impl Display for JrOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JR {}", self.reg)
    }
}

impl JrOp {
    pub(crate) fn to_jump(self, cpu: &Cpu) -> JOp {
        let dest = Addr(cpu.reg(self.reg));
        JOp { dest }
    }
}

/// call rs,ret=rd | jalr (rd,)rs(,rd) | pc=rs, rd=$+8 ;see caution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JalrOp {
    pub(crate) dest: RegisterId,
    pub(crate) ret: RegisterId,
}

impl const From<Op> for JalrOp {
    #[inline]
    fn from(value: Op) -> Self {
        JalrOp {
            dest: value.bits(21..26) as usize,
            ret: value.bits(11..16) as usize,
        }
    }
}

impl const From<JalrOp> for Op {
    #[inline]
    fn from(value: JalrOp) -> Self {
        Op(0)
            .with_primary(PrimaryOp::SPECIAL)
            .with_secondary(SecondaryOp::JALR)
            .set_bits(21..26, value.dest as u32)
            .set_bits(11..16, value.ret as u32)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn jalr(rd: RegisterId, rs: RegisterId) -> Op {
        JalrOp { dest: rs, ret: rd }.into()
    }
}

impl Display for JalrOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JALR {} {}", self.ret, self.dest)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{
        cpu::op::Op,
        memory::{KSEG0Addr, PhysAddr},
    };

    #[rstest]
    fn create_j_1(setup_tracing: ()) {
        let j = Op::j(KSEG0Addr::from_phys(0x0000_2000));
        assert_eq!(j.to_string(), "J 0x00002000");
    }

    #[rstest]
    fn create_jal(setup_tracing: ()) {
        let j = Op::jal(KSEG0Addr::from_phys(0x0000_2000));
        assert_eq!(j.to_string(), "JAL 0x00002000");
    }

    #[rstest]
    fn create_jr(setup_tracing: ()) {
        let j = Op::jr(8);
        assert_eq!(j.to_string(), "JR 8");
    }

    #[rstest]
    fn create_jalr(setup_tracing: ()) {
        let jalr = Op::jalr(8, 9);
        assert_eq!(jalr.to_string(), "JALR 8 9");
    }
}
