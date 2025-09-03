use std::fmt::Display;

use crate::{
    cpu::op::{Op, PrimaryOp},
    memory::{Addr, Address},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JOp {
    pub(crate) dest: Addr,
}

impl const From<Op> for JOp {
    #[inline]
    fn from(value: Op) -> Self {
        JOp {
            dest: Addr((value.bits(0..26) as u32) << 2),
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
}
