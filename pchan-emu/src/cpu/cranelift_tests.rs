use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;

use crate::{Emu, test_utils::emulator};

#[rstest]
fn basic_jit(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
    use crate::cranelift_bs::*;

    // Function signature: i32 fn()
    let mut sig = Signature::new(CallConv::SystemV);
    sig.returns.push(AbiParam::new(types::I32));

    let func_id = emulator
        .jit
        .module
        .declare_function("const_42", Linkage::Export, &sig)?;
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

    // Build function: return 42
    {
        let mut builder = FunctionBuilder::new(&mut func, &mut emulator.jit.fn_builder_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        let value = builder.ins().iconst(types::I32, 42);
        builder.ins().return_(&[value]);
        builder.finalize();
    }
    emulator.jit.ctx.func = func;
    // Create function
    emulator
        .jit
        .module
        .define_function(func_id, &mut emulator.jit.ctx)?;
    emulator.jit.module.clear_context(&mut emulator.jit.ctx);
    emulator.jit.module.finalize_definitions()?;

    let code_ptr = emulator.jit.module.get_finalized_function(func_id);
    let const_fn = unsafe { std::mem::transmute::<*const u8, fn() -> i32>(code_ptr) };
    tracing::info!("Result: {}", const_fn()); // Prints: Result: 42
    assert_eq!(const_fn(), 42);
    Ok(())
}
