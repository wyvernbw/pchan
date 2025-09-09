#![feature(duration_millis_float)]
#![allow(unused_variables)]

use pchan_emu::{
    Emu, JitSummary,
    cpu::{Cpu, ops::OpCode},
    memory::{KSEG0Addr, Memory},
};
use pchan_utils::setup_tracing;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pchan_emu::cpu::ops::prelude::*;

    setup_tracing();
    let mut emulator = Emu::default();

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

    let now = Instant::now();
    emulator.step_jit()?;
    let cold_elapsed = now.elapsed().as_millis_f64();

    let slice = &emulator.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
    assert_eq!(slice, &[0, 1, 2, 3]);

    let mut average = 0;
    for _ in 0..100 {
        let now = Instant::now();
        emulator.step_jit()?;
        average += now.elapsed().as_nanos();
    }
    let average = (average as f64) / 100.0;

    let slice = &emulator.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
    assert_eq!(slice, &[0, 1, 2, 3]);

    tracing::info!("cold run: {}ms", cold_elapsed);
    tracing::info!("hot run average across 100 runs: {}ns", average);

    Ok(())
}
