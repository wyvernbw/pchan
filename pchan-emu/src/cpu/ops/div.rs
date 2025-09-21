use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, derive_more::Display)]
#[display("div ${},${}", reg_str(self.rs), reg_str(self.rt))]
pub struct DIV {
    rs: u8,
    rt: u8,
}

impl DIV {
    pub const fn new(rs: u8, rt: u8) -> Self {
        Self { rs, rt }
    }
}

pub fn div(rs: u8, rt: u8) -> OpCode {
    DIV::new(rs, rt).into_opcode()
}

impl Op for DIV {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::DIV)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        let (rs, loadrs) = ctx.emit_get_register(self.rs);
        let (rt, loadrt) = ctx.emit_get_register(self.rt);
        let (res, div) = ctx.inst(|f| f.pure().Binary(Opcode::Sdiv, types::I32, rs, rt).0);
        let (remainder, srem) = ctx.inst(|f| f.pure().Binary(Opcode::Srem, types::I32, rs, rt).0);

        EmitSummary::builder()
            .hi(remainder)
            .lo(res)
            .instructions([now(loadrs), now(loadrt), now(div), now(srem)])
            .build(ctx.fn_builder)
    }
}

#[cfg(test)]
mod tests {
    use crate::dynarec::prelude::*;
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    #[rstest]
    #[case(4, 2, 2, 0)]
    #[case(6, 3, 2, 0)]
    #[case(5, 2, 2, 1)]
    #[case(i16::MIN + 1, i16::MAX, -1i32 as u32, 0)]
    pub fn div_1(
        setup_tracing: (),
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected_res: u32,
        #[case] expected_rem: u32,
    ) {
        use crate::Emu;

        let mut emu = Emu::default();
        emu.mem.write_many(
            0x0,
            &program([
                addiu(8, 0, a),
                addiu(9, 0, b),
                div(8, 9),
                nop(),
                OpCode(69420),
            ]),
        );

        let summary = emu.step_jit_summarize::<JitSummary>().unwrap();
        tracing::info!(?summary);

        assert_eq!((emu.cpu.hilo & ((1 << 32) - 1)) as u32, expected_res);
        assert_eq!((emu.cpu.hilo >> 32) as u32, expected_rem);
    }
}
