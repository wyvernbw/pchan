use std::fmt::Display;

use crate::cpu::{
    RegisterId,
    op::{Op, PrimaryOp, SecondaryOp},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubOp {
    pub(crate) header: SecondaryOp,
    pub(crate) rd: RegisterId,
    pub(crate) rs: RegisterId,
    pub(crate) rt: RegisterId,
}

impl const From<Op> for SubOp {
    #[inline]
    fn from(value: Op) -> Self {
        SubOp {
            header: value.secondary(),
            rs: value.bits(21..26) as usize,
            rt: value.bits(16..21) as usize,
            rd: value.bits(11..16) as usize,
        }
    }
}

impl const From<SubOp> for Op {
    #[inline]
    fn from(value: SubOp) -> Self {
        let SubOp { header, rs, rt, rd } = value;
        let i =
            0 | ((rs as u32) << 21) | ((rt as u32) << 16) | ((rd as u32) << 11) | (header as u32);
        Self(i)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn sub(rd: RegisterId, rs: RegisterId, rt: RegisterId) -> Op {
        SubOp {
            rd,
            rs,
            rt,
            header: SecondaryOp::SUB,
        }
        .into()
    }
    #[inline]
    pub(crate) const fn subu(rd: RegisterId, rs: RegisterId, rt: RegisterId) -> Op {
        SubOp {
            rd,
            rs,
            rt,
            header: SecondaryOp::SUBU,
        }
        .into()
    }
}

impl Display for SubOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {} {} {}", self.header, self.rd, self.rs, self.rt)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::cpu::op::Op;

    #[rstest]
    fn test_load_op_creation(setup_tracing: ()) {
        let op = Op::subu(8, 9, 4);
        assert_eq!(op.to_string(), "SUBU 8 9 4");
    }
}
