use std::fmt::Display;

use crate::cpu::{
    RegisterId,
    op::{Op, PrimaryOp, SecondaryOp},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AddOp {
    pub(crate) header: SecondaryOp,
    pub(crate) rd: RegisterId,
    pub(crate) rs: RegisterId,
    pub(crate) rt: RegisterId,
}

impl const From<Op> for AddOp {
    #[inline]
    fn from(value: Op) -> Self {
        AddOp {
            header: value.secondary(),
            rs: value.bits(21..26) as usize,
            rt: value.bits(16..21) as usize,
            rd: value.bits(11..16) as usize,
        }
    }
}

impl const From<AddOp> for Op {
    #[inline]
    fn from(value: AddOp) -> Self {
        let AddOp { header, rs, rt, rd } = value;
        let i =
            0 | ((rs as u32) << 21) | ((rt as u32) << 16) | ((rd as u32) << 11) | (header as u32);
        Self(i)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn add(rd: RegisterId, rs: RegisterId, rt: RegisterId) -> Op {
        AddOp {
            rd,
            rs,
            rt,
            header: SecondaryOp::ADD,
        }
        .into()
    }
    #[inline]
    pub(crate) const fn addu(rd: RegisterId, rs: RegisterId, rt: RegisterId) -> Op {
        AddOp {
            rd,
            rs,
            rt,
            header: SecondaryOp::ADDU,
        }
        .into()
    }
}

impl Display for AddOp {
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
        let op = Op::add(8, 9, 4);
        assert_eq!(op.to_string(), "ADD 8 9 4");
    }
}
