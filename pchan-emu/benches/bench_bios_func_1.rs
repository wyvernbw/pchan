use color_eyre::eyre::ContextCompat;
use criterion::*;
use pchan_emu::{
    Emu,
    dynarec::pipeline::EmuDynarecPipeline,
    jit::{BlockFn, JIT},
};

fn get_function(emu: &mut Emu, jit: &mut JIT) -> color_eyre::Result<BlockFn> {
    emu.jit_cache.clear_cache();
    emu.cpu.pc = 0xbfc0_0000;
    jit.ctx.want_disasm = true;
    EmuDynarecPipeline::iter(emu, jit)
        .take(5)
        .find_map(|pipe| {
            if let EmuDynarecPipeline::Emitted { function, .. } = pipe {
                Some(function)
            } else {
                None
            }
        })
        .wrap_err("could not compile")
}

fn bench(c: &mut Criterion) {
    let mut emu = Emu::default();
    emu.load_bios().unwrap();
    let mut jit = JIT::default();
    let func = get_function(&mut emu, &mut jit).unwrap();
    println!("compiled to:");
    println!("{}", func.func);
    let disasm = jit
        .ctx
        .compiled_code()
        .expect("no compiled code.")
        .vcode
        .as_ref()
        .expect("no disassembled code");
    println!("compiled to:");
    println!("{}", disasm);
    c.bench_function("func_1", |b| {
        b.iter(|| {
            let _: () = func.call_block((&mut emu, false));
            black_box(());
        })
    });
    println!("ran for {} cycles.", emu.cpu.d_clock);
    let op_count = disasm.lines().count();
    println!("ran ~{op_count} instructions.");

    c.bench_function("compile_and_run_func_1", |b| {
        b.iter(|| {
            let func = get_function(&mut emu, &mut jit).unwrap();
            let _: () = func.call_block((&mut emu, false));
            black_box(());
        });
    });
    println!("ran for {} cycles.", emu.cpu.d_clock);
    println!("ran ~{op_count} instructions.");
}

criterion_group!(benches, bench);
criterion_main!(benches);
