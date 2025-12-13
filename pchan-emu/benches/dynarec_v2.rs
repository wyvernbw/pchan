use std::cell::RefCell;

use criterion::{Criterion, criterion_group, criterion_main};
use pchan_emu::{Emu, cpu::ops::OpCode, dynarec::prelude::addiu, dynarec_v2::PipelineV2};
use pchan_utils::setup_tracing;

fn dynarec_v2_test_50_adds_complete(c: &mut Criterion) {
    tracing::info!(
        r#"running entire pipeline:
        - fetch & compile
        - execute
        - cache"#
    );
    let emu = dynarec_v2_50_adds_setup();

    c.bench_function("add_50_complete", |b| {
        b.iter_batched(
            || {
                _ = emu.borrow_mut().dynarec_cache.take(0x0);
                PipelineV2::new(&emu.borrow())
            },
            |mut pipe| {
                for _ in 0..3 {
                    pipe = pipe.step(&mut emu.borrow_mut()).unwrap();
                }
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

fn dynarec_v2_test_50_adds_execute(c: &mut Criterion) {
    tracing::info!(
        r#"running pipeline subset:
        - execute
        "#
    );
    let emu = dynarec_v2_50_adds_setup();
    let mut pipe = PipelineV2::new(&emu.borrow());
    pipe = pipe.step(&mut emu.borrow_mut()).unwrap();
    let PipelineV2::Compiled { func, .. } = pipe else {
        panic!("wrong stage");
    };

    c.bench_function("add_50_execute", |b| {
        b.iter(|| {
            func(&mut emu.borrow_mut(), false);
        });
    });
}

fn dynarec_v2_50_adds_setup() -> &'static RefCell<Emu> {
    let instruction_count = 50;
    setup_tracing();
    let emu: &'static _ = Box::leak(Box::new(RefCell::new(Emu::default())));
    let mut last_addr = 0x0;
    for addr in (0x0..).step_by(4).take(instruction_count) {
        emu.borrow_mut().write(addr, addiu(8, 8, 1));
        last_addr = addr;
    }
    emu.borrow_mut().write(last_addr + 4, OpCode::HALT_FIELDS);
    emu
}

criterion_group!(
    benches,
    dynarec_v2_test_50_adds_execute,
    dynarec_v2_test_50_adds_complete
);
criterion_main!(benches);
