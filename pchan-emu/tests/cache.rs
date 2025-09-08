#![feature(duration_millis_float)]
#![allow(unused_variables)]

mod common;

use pchan_emu::{
    Emu, JitSummary,
    cpu::{Cpu, ops::OpCode},
    memory::{KSEG0Addr, Memory},
};
use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;
use std::time::Instant;

use crate::common::emulator;

#[rstest]
fn block_compile_cache(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
    use pchan_emu::cpu::ops::prelude::*;

    let init = |emulator: &mut Emu| {
        emulator.mem = Memory::default();
        emulator.cpu = Cpu::default();
        let program = [
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
            .write_array(KSEG0Addr::from_phys(emulator.cpu.pc as u32), &program);
    };

    init(&mut emulator);

    let now = Instant::now();
    emulator.step_jit()?;
    let cold_elapsed = now.elapsed().as_millis_f64();

    let slice = &emulator.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
    assert_eq!(slice, &[0, 1, 2, 3]);

    let mut average = 0.0;
    for _ in 0..100 {
        init(&mut emulator);
        let now = Instant::now();
        emulator.step_jit()?;
        average += now.elapsed().as_millis_f64();
    }
    average /= 100.0;

    let slice = &emulator.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
    assert_eq!(slice, &[0, 1, 2, 3]);

    tracing::info!("cold run: {}ms", cold_elapsed);
    tracing::info!("hot run average across 100 runs: {}ms", average);

    Ok(())
}

#[rstest]
fn register_cache_test(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
    use pchan_emu::cpu::ops::prelude::*;

    let program = [
        addiu(9, 0, 69), // 1x iconst.i64 + 1x iadd_imm + 1x store = 3 ops
        addiu(8, 9, 69), // 1x iadd_imm + 1x store = 2 ops
        addiu(8, 9, 69), // 1x iadd_imm = 1 op
        addiu(8, 9, 69), // ...
        addiu(8, 9, 69),
        addiu(8, 9, 69),
        addiu(8, 9, 69), // +1 return
        OpCode(69420),
    ];

    emulator
        .mem
        .write_array(KSEG0Addr::from_phys(emulator.cpu.pc as u32), &program);

    let summary = emulator.step_jit_summarize::<JitSummary>()?;
    let op_count = summary.function.unwrap().dfg.num_insts();

    assert!(op_count <= 7 + 2 + 1 + 1);

    Ok(())
}
