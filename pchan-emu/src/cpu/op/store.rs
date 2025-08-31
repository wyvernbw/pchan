use std::fmt::Display;

use crate::cpu::{
    RegisterId,
    op::{Op, PrimaryOp},
};

/// IR for the store instruction, pretty much structurally equivalent to Load
/// only the semantics are different
///
/// ```asm
/// movb  [imm+rs],rt  sb  rt,imm(rs)    [imm+rs]=(rt AND FFh)   ;store 8bit
/// movh  [imm+rs],rt  sh  rt,imm(rs)    [imm+rs]=(rt AND FFFFh) ;store 16bit
/// mov   [imm+rs],rt  sw  rt,imm(rs)    [imm+rs]=rt             ;store 32bit
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StoreOp {
    /// type of store
    pub(crate) header: PrimaryOp,
    /// register of the address to store at
    pub(crate) rs: RegisterId,
    /// register of the value to store
    pub(crate) rt: RegisterId,
    /// offset applied to `rs`
    pub(crate) imm: i16,
}

impl Display for StoreOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {} {} {}", self.header, self.rt, self.rs, self.imm)
    }
}

impl const From<Op> for StoreOp {
    #[inline]
    fn from(value: Op) -> Self {
        StoreOp {
            header: value.primary(),
            rs: value.bits(21..26) as usize,
            rt: value.bits(16..21) as usize,
            imm: value.bits(0..16) as i16,
        }
    }
}

impl const From<StoreOp> for Op {
    #[inline]
    fn from(value: StoreOp) -> Self {
        let StoreOp {
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
    pub(crate) const fn sb(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        StoreOp {
            header: PrimaryOp::SB,
            rs,
            rt,
            imm,
        }
        .into()
    }
    #[inline]
    pub(crate) const fn sh(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        StoreOp {
            header: PrimaryOp::SH,
            rs,
            rt,
            imm,
        }
        .into()
    }
    #[inline]
    pub(crate) const fn sw(rt: RegisterId, rs: RegisterId, imm: i16) -> Op {
        StoreOp {
            header: PrimaryOp::SW,
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
    fn test_store_op_creation(setup_tracing: ()) {
        let op = Op::sb(8, 9, 4);
        assert_eq!(op.to_string(), "SB 8 9 4");

        let op = Op::sh(12, 3, 4);
        assert_eq!(op.to_string(), "SH 12 3 4");

        let op = Op::sw(5, 6, 4);
        assert_eq!(op.to_string(), "SW 5 6 4");
    }
}
