#![feature(duration_millis_float)]

mod common;

use pchan_emu::{Emu, cpu::ops::OpCode, memory::KSEG0Addr};
use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;
use std::time::Instant;

use crate::common::emulator;

#[rstest]
fn block_compile_cache(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
    use pchan_emu::cpu::ops::prelude::*;

    let program = [
        addiu(8, 0, 32),
        j(KSEG0Addr::from_phys(0x0000_2000).as_u32() as i32),
        addiu(10, 0, 32),
    ];

    let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

    emulator
        .mem
        .write_array(KSEG0Addr::from_phys(emulator.cpu.pc as u32), &program);
    emulator
        .mem
        .write_array(KSEG0Addr::from_phys(0x0000_2000), &function);

    let now = Instant::now();
    emulator.advance_jit()?;
    let cold_elapsed = now.elapsed().as_millis_f64();

    assert_eq!(emulator.cpu.gpr[9], 69);
    assert_eq!(emulator.cpu.gpr[10], 32);

    emulator.cpu.pc = 0;

    let mut average = 0.0;
    for _ in 0..100 {
        let now = Instant::now();
        emulator.advance_jit()?;
        average += now.elapsed().as_millis_f64();
    }
    average /= 100.0;

    assert_eq!(emulator.cpu.gpr[9], 69);
    assert_eq!(emulator.cpu.gpr[10], 32);

    tracing::info!("cold run: {}ms", cold_elapsed);
    tracing::info!("hot run average across 100 runs: {}ms", average);

    Ok(())
}
