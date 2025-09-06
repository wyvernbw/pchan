use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::cranelift_bs::*;

use super::{Opcode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub(crate) struct LB {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub(crate) fn lb(rt: usize, rs: usize, imm: i16) -> ops::Opcode {
    LB { rt, rs, imm }.into_opcode()
}

impl LB {
    pub(crate) fn try_from_opcode(opcode: Opcode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LB)?;
        Ok(LB {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LB {
    fn emit_ir(&self, mut state: super::EmitParams<'_, '_>) -> Option<EmitSummary> {
        // get pointer to memory passed as argument to the function
        let mem_ptr = state.memory();

        // get cached register if possible, otherwise load it in
        let rs = state.emit_get_register(self.rs);
        let mem_ptr = state.fn_builder.ins().iadd(mem_ptr, rs);

        let rt =
            state
                .fn_builder
                .ins()
                .sload8(types::I64, MemFlags::new(), mem_ptr, self.imm as i32);
        Some(EmitSummary {
            register_updates: vec![(self.rt, rt)].into_boxed_slice(),
        })
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::Opcode {
        ops::Opcode::default()
            .with_primary(PrimeOp::LB)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::{
            JIT,
            ops::{self, EmitParams, Op, lb::LB},
        },
        memory::KSEG0Addr,
        test_utils::emulator,
    };

    #[rstest]
    fn test_lb(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cranelift_bs::*;

        let sig = emulator.jit.create_signature();
        let ptr_type = emulator.jit.pointer_type();

        let func_id = emulator
            .jit
            .module
            .declare_function("test_lb", Linkage::Hidden, &sig)?;
        let mut func = Function::with_name_signature(UserFuncName::user(0, 1), sig);

        let mut fn_builder = FunctionBuilder::new(&mut func, &mut emulator.jit.fn_builder_ctx);
        let block = JIT::init_block(&mut fn_builder);
        let lb = LB {
            rs: 8,
            rt: 9,
            imm: 16,
        };
        let params = EmitParams {
            ptr_type,
            fn_builder: &mut fn_builder,
            registers: &[None; 32],
            pc: 0,
            block,
        };
        let summary = lb.emit_ir(params).unwrap();
        tracing::info!(?summary);
        JIT::emit_updates()
            .builder(&mut fn_builder)
            .block(block)
            .summary(&summary)
            .call();
        fn_builder.ins().return_(&[]);
        fn_builder.finalize();

        emulator.jit.ctx.func = func;
        emulator
            .jit
            .module
            .define_function(func_id, &mut emulator.jit.ctx)?;

        emulator.jit.module.clear_context(&mut emulator.jit.ctx);
        emulator.jit.module.finalize_definitions()?;

        let test_lb = emulator.jit.get_func(func_id);

        emulator.cpu.gpr[8] = 4; // address of element at index 4 in memory
        emulator.mem.as_mut()[4 + 16] = 69;

        tracing::info!(%emulator.cpu);
        tracing::info!("v2(rs) = {}", emulator.cpu.gpr[8]);
        tracing::info!("memory[4+16] = {}", emulator.mem.as_ref()[20]);
        test_lb(&mut emulator.cpu, &mut emulator.mem);

        tracing::info!(%emulator.cpu);
        assert_eq!(emulator.cpu.gpr[9], 69);

        Ok(())
    }

    #[rstest]
    pub fn test_lb_sign_extension(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator.mem.write(KSEG0Addr::from_phys(32), 0xFFu8);
        emulator.mem.write(KSEG0Addr::from_phys(33), 0x7Fu8);
        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lb(8, 9, 0), lb(10, 9, 1), ops::Opcode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.advance_jit()?;

        // 0xFF should be sign-extended to 0xFFFFFFFFFFFFFFFF
        assert_eq!(emulator.cpu.gpr[8], 0xFFFFFFFFFFFFFFFF);

        // 0x7F should be sign-extended to 0x7F
        assert_eq!(emulator.cpu.gpr[10], 0x7F);

        Ok(())
    }
}
