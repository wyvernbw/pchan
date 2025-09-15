#![feature(slice_as_array)]
#![allow(unused_variables)]

use pchan_emu::{
    Emu,
    dynarec::JitSummary,
    memory::{KSEG0Addr, map_physical},
};
use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;

#[rstest]
// ram
#[case(0x0000_0000)]
#[case(0x8000_0000)]
#[case(0xA000_0000)]
// expansion 1
#[case(0x1F00_0000)]
#[case(0x9F00_0000)]
#[case(0xBF00_0000)]
// scratchpad
#[case(0x1F80_0000)]
#[case(0x9F80_0000)]
// io ports
#[case(0x1F80_1000)]
#[case(0x9F80_1000)]
#[case(0xBF80_1000)]
fn write_to_address(setup_tracing: (), #[case] base: u32) -> color_eyre::Result<()> {
    use pchan_emu::{cpu::ops::prelude::*, memory::PhysAddr};

    let mut emu = Emu::default();
    emu.mem.write_array(
        KSEG0Addr::from_phys(emu.cpu.pc),
        &[
            addiu(8, 0, 0),
            addiu(10, 0, 256),
            lui(9, (base >> 16) as i16),
            ori(9, 9, (base & 0x0000_FFFF) as i16),
            addiu(11, 0, 69),
            sw(8, 9, 0),
            addiu(8, 8, 4),
            bne(8, 10, -12),
            nop(),
            OpCode(69420),
        ],
    );
    let summary = emu.step_jit_summarize::<JitSummary>()?;
    tracing::info!(?summary.function);

    // assert_eq!(emu.cpu.gpr[9], base);
    assert_eq!(emu.cpu.gpr[11], 69);
    let base = map_physical(PhysAddr(base & 0x1FFF_FFFF));
    for i in 0..256 {
        let idx = base + i * 4;
        let idx = idx as usize;
        let value = &emu.mem.as_ref()[idx..(idx + 4)];
        let value = *value.as_array().unwrap();
        let value = u32::from_le_bytes(value);
        assert_eq!(value, i * 4);
    }

    tracing::info!(?summary.function);

    Ok(())
}
