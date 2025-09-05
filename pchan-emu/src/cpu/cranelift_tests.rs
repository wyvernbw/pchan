use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::{fixture, rstest};

use crate::{
    cpu::{Cpu, JIT},
    memory::Memory,
};

#[fixture]
fn cpu() -> Cpu {
    Cpu::default()
}
#[fixture]
fn mem() -> Memory {
    Memory::default()
}
#[fixture]
fn jit() -> JIT {
    JIT::default()
}

#[rstest]
fn basic_jit(setup_tracing: (), mut cpu: Cpu, mut jit: JIT) -> color_eyre::Result<()> {
    use crate::cranelift_bs::*;

    // Function signature: i32 fn()
    let mut sig = Signature::new(CallConv::SystemV);
    sig.returns.push(AbiParam::new(types::I32));

    let func_id = jit
        .module
        .declare_function("const_42", Linkage::Export, &sig)?;
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

    // Build function: return 42
    {
        let mut builder = FunctionBuilder::new(&mut func, &mut jit.fn_builder_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        let value = builder.ins().iconst(types::I32, 42);
        builder.ins().return_(&[value]);
        builder.finalize();
    }
    jit.ctx.func = func;
    // Create function
    jit.module.define_function(func_id, &mut jit.ctx)?;
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions()?;

    let code_ptr = jit.module.get_finalized_function(func_id);
    let const_fn = unsafe { std::mem::transmute::<*const u8, fn() -> i32>(code_ptr) };
    tracing::info!("Result: {}", const_fn()); // Prints: Result: 42
    assert_eq!(const_fn(), 42);
    Ok(())
}

#[rstest]
fn basic_adder(
    setup_tracing: (),
    mut cpu: Cpu,
    mut mem: Memory,
    mut jit: JIT,
) -> color_eyre::Result<()> {
    use crate::cranelift_bs::*;

    let sig = jit.create_signature();

    let func_id = jit
        .module
        .declare_function("adder", Linkage::Export, &sig)?;
    let mut func = Function::with_name_signature(UserFuncName::user(0, 1), sig);

    {
        let mut builder = FunctionBuilder::new(&mut func, &mut jit.fn_builder_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);

        let a = JIT::emit_load_reg()
            .builder(&mut builder)
            .block(block)
            .idx(9)
            .call();
        let b = JIT::emit_load_reg()
            .builder(&mut builder)
            .block(block)
            .idx(10)
            .call();

        // x + y
        let sum = builder.ins().iadd(a, b);
        JIT::emit_store_reg()
            .builder(&mut builder)
            .block(block)
            .idx(11)
            .value(sum)
            .call();
        builder.ins().return_(&[]);

        builder.finalize();
    }

    jit.ctx.func = func;
    jit.module.define_function(func_id, &mut jit.ctx)?;

    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions()?;

    let adder = jit.get_func(func_id);

    cpu.gpr[9] = 69;
    cpu.gpr[10] = 42;

    adder(&cpu, &mem);

    assert_eq!(cpu.gpr[11], cpu.gpr[9] + cpu.gpr[10]);

    Ok(())
}
