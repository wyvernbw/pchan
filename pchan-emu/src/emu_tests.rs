use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;

use crate::{Emu, cpu::ops::OpCode, memory::KSEG0Addr, test_utils::emulator};

#[rstest]
fn block_compile_cache(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
    use crate::cpu::ops::prelude::*;

    let program = [
        addiu(8, 0, 32),
        j(KSEG0Addr::from_phys(0x0000_2000).as_u32() as i32),
        addiu(10, 0, 32),
    ];

    let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

    emulator
        .mem
        .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), program);
    emulator
        .mem
        .write_all(KSEG0Addr::from_phys(0x0000_2000), function);

    emulator.advance_jit()?;

    assert_eq!(emulator.cpu.gpr[9], 69);
    assert_eq!(emulator.cpu.gpr[10], 32);

    emulator.cpu.pc = 0;

    emulator.advance_jit()?;

    assert_eq!(emulator.cpu.gpr[9], 69);
    assert_eq!(emulator.cpu.gpr[10], 32);

    Ok(())
}
