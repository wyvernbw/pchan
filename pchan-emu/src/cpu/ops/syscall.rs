use crate::cpu::ops::OpCode;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, derive_more::Display)]
#[display("syscall")]
pub struct Syscall;

impl From<Syscall> for OpCode {
    fn from(_: Syscall) -> Self {
        let mut op = OpCode::default();
        op.set_funct(0xc);
        op
    }
}
