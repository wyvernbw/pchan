use std::fmt::Display;

use crate::dynarec::prelude::*;

use super::{EmitCtx, OpCode, PrimeOp};
use crate::load;

#[derive(Debug, Clone, Copy)]
pub struct LB {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn lb(rt: usize, rs: usize, imm: i16) -> OpCode {
    LB { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LB {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LB)?;
        Ok(LB {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LB {
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, Opcode::Sload8)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::LB)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

impl Display for LB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lb ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::ops::{self},
        memory::KSEG0Addr,
        test_utils::emulator,
    };

    #[rstest]
    pub fn test_lb_sign_extension(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator.mem.write(KSEG0Addr::from_phys(32), 0xFFu8);
        emulator.mem.write(KSEG0Addr::from_phys(33), 0x7Fu8);
        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lb(8, 9, 0), lb(10, 9, 1), nop(), ops::OpCode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], -1i32 as u32);

        // 0x7F should be sign-extended to 0x7F
        assert_eq!(emulator.cpu.gpr[10], 0x7F);

        Ok(())
    }
}

#[macro_export]
macro_rules! load {
    ($self: expr, $ctx: expr, $opcode: expr) => {{
        use cranelift::codegen::ir::immediates::Offset32;
        use $crate::dynarec::prelude::*;

        // get pointer to memory passed as argument to the function
        let mem_ptr = $ctx.memory();
        let ptr_type = $ctx.ptr_type;
        let (rs, loadreg) = $ctx.emit_get_register($self.rs);
        let (rs, mapaddr) = $ctx.emit_map_address_to_host(rs);

        let (mem_ptr, iadd0) = $ctx.inst(|f| f.ins().Binary(Opcode::Iadd, ptr_type, mem_ptr, rs).0);

        let (rt, sload8) = $ctx.inst(|f| {
            f.ins()
                .Load(
                    $opcode,
                    types::I32,
                    MemFlags::new(),
                    Offset32::new($self.imm as i32),
                    mem_ptr,
                )
                .0
        });

        EmitSummary::builder()
            .instructions(
                [
                    [now(loadreg)].as_slice(),
                    mapaddr.map(now).as_slice(),
                    [now(iadd0), delayed(1, sload8)].as_slice(),
                ]
                .concat(),
            )
            .register_updates([($self.rt, rt)])
            .build($ctx.fn_builder)
    }};
}
