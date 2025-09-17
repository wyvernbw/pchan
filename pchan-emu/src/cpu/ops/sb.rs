use crate::dynarec::prelude::*;
use std::fmt::Display;

use super::{EmitCtx, OpCode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub struct SB {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn sb(rt: usize, rs: usize, imm: i16) -> OpCode {
    SB { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for SB {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::SB)?;
        Ok(SB {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for SB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sb ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for SB {
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        store!(self, ctx, Opcode::Istore8)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SB)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    pub fn test_sb(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0), [sb(9, 8, 0), OpCode(69420)]);

        emulator.cpu.gpr[8] = 32; // base register
        emulator.cpu.gpr[9] = 69;

        emulator.step_jit()?;

        assert_eq!(emulator.mem.read::<u8>(KSEG0Addr::from_phys(32)), 69);

        Ok(())
    }
}

#[macro_export]
macro_rules! store {
    ($self:expr, $ctx:expr, $opcode:expr) => {{
        use cranelift::codegen::ir::immediates::Offset32;
        use $crate::dynarec::prelude::*;

        // get pointer to memory passed as argument to the function
        let mem_ptr = $ctx.memory();
        let ptr_type = $ctx.ptr_type;

        // get cached register if possible, otherwise load it in
        let (rs, loadrs) = $ctx.emit_get_register($self.rs);
        let (rs, mapaddr) = $ctx.emit_map_address_to_host(rs);
        let (rt, loadrt) = $ctx.emit_get_register($self.rt);
        let (mem_ptr, iadd) = $ctx.inst(|f| f.pure().Binary(Opcode::Iadd, ptr_type, mem_ptr, rs).0);

        let store = $ctx
            .fn_builder
            .pure()
            .Store(
                $opcode,
                types::I32,
                MemFlags::new(),
                Offset32::new($self.imm.into()),
                rt,
                mem_ptr,
            )
            .0;

        EmitSummary::builder()
            .instructions(
                [
                    [now(loadrs)].as_slice(),
                    mapaddr.map(now).as_slice(),
                    [now(loadrt), now(iadd), delayed(1, store)].as_slice(),
                ]
                .concat(),
            )
            .build($ctx.fn_builder)
    }};
}
