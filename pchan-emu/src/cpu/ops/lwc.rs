use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LWCn {
    pub cop: u8,
    pub rt: u8,
    pub rs: u8,
    pub imm: i16,
}

impl LWCn {
    pub const fn new(cop: u8, rt: u8, rs: u8, imm: i16) -> Self {
        Self { cop, rt, rs, imm }
    }
}

impl Display for LWCn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lwc{} ${} ${} {}",
            self.cop,
            reg_str(self.rt),
            reg_str(self.rs),
            hex(self.imm)
        )
    }
}

impl Op for LWCn {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(match self.cop {
                0 => PrimeOp::LWC0,
                1 => PrimeOp::LWC1,
                2 => PrimeOp::LWC2,
                3 => PrimeOp::LWC3,
                _ => panic!("invalid coprocessor"),
            })
            .set_bits(0..16, self.imm as i32 as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        tracing::info!("LWC EMITTING!");
        let mem_ptr = ctx.memory();
        let cpu_ptr = ctx.cpu();
        let (rs, loadreg) = ctx.emit_get_register(self.rs);
        let (rs, addinst) = ctx.inst(|f| {
            f.pure()
                .BinaryImm64(Opcode::IaddImm, types::I32, Imm64::new(self.imm as i64), rs)
                .0
        });
        let (rt, readinst) = ctx.inst(|f| {
            f.pure()
                .call(ctx.func_ref_table.read32, &[mem_ptr, cpu_ptr, rs])
        });
        let store = ctx.emit_store_cop_register(self.cop, self.rt as usize, rt);

        EmitSummary::builder()
            .instructions([now(loadreg), now(addinst), now(readinst), delayed(1, store)])
            .build(ctx.fn_builder)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }
}

pub fn lwc(n: u8) -> impl Fn(u8, u8, i16) -> OpCode {
    move |rt, rs, imm| LWCn::new(n, rt, rs, imm).into_opcode()
}

#[cfg(test)]
mod tests {
    use crate::{dynarec::prelude::*, jit::JIT, test_utils::jit};
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn lwc_1(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        tracing::info!(?emulator.cpu);
        emulator.write_many(
            0x0,
            &program([
                addiu(9, 0, 69),
                sw(9, 0, 0x0000_1000),
                nop(),
                lwc(1)(5, 0, 0x0000_1000),
                nop(),
                OpCode(69420),
            ]),
        );

        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;

        tracing::info!(%summary.decoded_ops);
        tracing::info!(?summary.function);
        tracing::info!(?emulator.cpu);

        assert_eq!(emulator.read::<u32, ext::NoExt>(0x0000_1000), 69);
        assert_eq!(emulator.cpu.cop1.reg[5], 69);
        Ok(())
    }
}
