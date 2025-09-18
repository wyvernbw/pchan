#[macro_export]
macro_rules! load {
    ($self:expr, $ctx:expr, $func:ident) => {{
        use $crate::dynarec::prelude::*;

        let mem_ptr = $ctx.memory();
        // let ptr_type = $ctx.ptr_type;
        let (rs, loadreg) = $ctx.emit_get_register($self.rs);
        let (rt, readinst) =
            $ctx.inst(|f| f.pure().call($ctx.func_ref_table.$func, &[mem_ptr, rs]));

        EmitSummary::builder()
            .instructions([now(loadreg), delayed(1, readinst)])
            .register_updates([($self.rt, updtdelay(1, rt))])
            .build($ctx.fn_builder)
    }};
}
