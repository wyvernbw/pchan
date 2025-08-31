use crate::cpu::{
    RegisterId,
    op::{Op, PrimaryOp},
};

/// IR for load operations. on write, it gets encoded as a u32
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoadOp {
    pub(crate) header: PrimaryOp,
    pub(crate) rs: RegisterId,
    pub(crate) rt: RegisterId,
    pub(crate) imm: i16,
}

impl const From<Op> for LoadOp {
    #[inline]
    fn from(value: Op) -> Self {
        LoadOp {
            header: value.primary(),
            rs: value.bits(21..26) as usize,
            rt: value.bits(16..21) as usize,
            imm: value.bits(0..16) as i16,
        }
    }
}

impl const From<LoadOp> for Op {
    #[inline]
    fn from(value: LoadOp) -> Self {
        let LoadOp {
            header,
            rs,
            rt,
            imm,
        } = value;
        let i = ((header as u32) << 26) | ((rs as u32) << 21) | ((rt as u32) << 16) | (imm as u32);
        Self(i)
    }
}

impl Op {
    #[inline]
    pub(crate) const fn load_op(&self) -> LoadOp {
        LoadOp {
            rs: self.bits(21..26) as usize,
            rt: self.bits(16..21) as usize,
            imm: self.bits(0..16) as i16,
            header: self.primary(),
        }
    }

    #[inline]
    pub(crate) const fn lb(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LB,
            rs,
            rt,
            imm,
        }
        .into()
    }
    #[inline]
    pub(crate) const fn lbu(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LBU,
            rs,
            rt,
            imm,
        }
        .into()
    }

    #[inline]
    pub(crate) const fn lh(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LH,
            rs,
            rt,
            imm,
        }
        .into()
    }

    #[inline]
    pub(crate) const fn lhu(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LHU,
            rs,
            rt,
            imm,
        }
        .into()
    }

    #[inline]
    pub(crate) const fn lw(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LW,
            rs,
            rt,
            imm,
        }
        .into()
    }

    #[inline]
    pub(crate) const fn lwl(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LWL,
            rs,
            rt,
            imm,
        }
        .into()
    }

    #[inline]
    pub(crate) const fn lwr(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        LoadOp {
            header: PrimaryOp::LWR,
            rs,
            rt,
            imm,
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::cpu::op::Op;

    #[rstest]
    fn test_load_op_creation(setup_tracing: ()) {
        let op = Op::lb(8, 9, 4);
        assert_eq!(op.to_string(), "LB 8 9 4");
    }
}
