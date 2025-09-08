use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;
use tracing::instrument;

use crate::cpu::{
    REG_STR,
    ops::{
        BoundaryType, EmitParams, EmitSummary, MipsOffset, Op, OpCode, PrimeOp, TryFromOpcodeErr,
    },
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct BEQ {
    rs: usize,
    rt: usize,
    imm: i32,
}

#[inline]
pub fn beq(rs: usize, rt: usize, dest: i32) -> OpCode {
    BEQ { rs, rt, imm: dest }.into_opcode()
}

impl BEQ {
    pub fn try_from_opcode(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::BEQ)?;
        Ok(BEQ {
            rs: (opcode.bits(21..26)) as usize,
            rt: (opcode.bits(16..21)) as usize,
            imm: (opcode.bits(0..16) as i32) << 2,
        })
    }
}

impl Display for BEQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "beq ${} ${} 0x{:08X}",
            REG_STR[self.rs], REG_STR[self.rt], self.imm
        )
    }
}

impl Op for BEQ {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::BlockSplit {
            lhs: MipsOffset::Relative(self.imm + 4),
            rhs: MipsOffset::Relative(4),
        })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::BEQ)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm >> 2) as i16 as u32)
    }

    #[instrument("beq", skip_all)]
    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;

        let rs = state.emit_get_register(fn_builder, self.rs);
        let rt = state.emit_get_register(fn_builder, self.rt);

        let then_block = state.next_at(0);
        let else_block = state.next_at(1);

        let then_params = state.out_params(then_block.clif_block, fn_builder);
        let else_params = state.out_params(else_block.clif_block, fn_builder);

        let cond = fn_builder.ins().icmp(IntCC::Equal, rs, rt);
        tracing::debug!(
            "branch: then={:?}({} deps) else={:?}({} deps)",
            then_block.clif_block,
            then_params.len(),
            else_block.clif_block,
            else_params.len()
        );

        fn_builder.ins().brif(
            cond,
            then_block.clif_block,
            &then_params,
            else_block.clif_block,
            &else_params,
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, cpu::ops::OpCode, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn beq_basic_loop(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let func = [
            addiu(8, 0, 0),           // ;  0 $t0 = 0
            addiu(10, 0, 4),          // ;  4 $t2 = 4
            addiu(9, 8, 0x0000_2000), // ;  8 calculate address $t1 = $t0 + 0x0000_2000
            sb(8, 9, 0),              // ; 12 store $i at $t1
            beq(8, 10, 16),           // ; 16 if $t0=$t2(4) jump by 16 to reach 36
            nop(),                    // ; 20
            addiu(8, 8, 1),           // ; 24 $t0 = $t0 + 1
            nop(),                    // ; 28
            j(8),                     // ; 32 jump to 8 (return to beginning of loop)
            nop(),                    // ; 36
            nop(),                    // ; 40
            OpCode(69420),            // ; 44 halt
        ];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), func);

        emulator.step_jit()?;

        let slice = &emulator.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
        assert_eq!(slice, &[0, 1, 2, 3]);

        Ok(())
    }
}
