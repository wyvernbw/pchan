#[macro_export]
macro_rules! store {
    ($self:expr, $ctx:expr, $func:ident) => {{
        use $crate::dynarec::prelude::*;

        let mem_ptr = $ctx.memory();
        let (rs, loadrs) = $ctx.emit_get_register($self.rs);
        let (rt, loadrt) = $ctx.emit_get_register($self.rt);
        let (address, add) = $ctx.inst(|f| {
            f.pure()
                .BinaryImm64(
                    Opcode::IaddImm,
                    types::I32,
                    Imm64::new($self.imm as i64),
                    rs,
                )
                .0
        });
        let storeinst = $ctx
            .fn_builder
            .pure()
            .call($ctx.func_ref_table.$func, &[mem_ptr, address, rt]);
        EmitSummary::builder()
            .instructions([now(loadrs), now(loadrt), now(add), delayed(1, storeinst)])
            .build($ctx.fn_builder)
    }};
}
