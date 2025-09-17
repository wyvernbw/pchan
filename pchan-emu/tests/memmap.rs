#![feature(slice_as_array)]
#![allow(unused_variables)]

use pchan_emu::{
    Emu,
    dynarec::JitSummary,
    memory::{KSEG0Addr, KSEG1Addr, lookup_phys, map_physical},
};
use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;

#[rstest]
// ram
#[case::kuseg_ram(0x0000_0000)]
#[case::kseg0_ram(0x8000_0000)]
#[case::kseg1_ram(0xA000_0000)]
// expansion 1
#[case::kuseg_exp1(0x1F00_0000)]
#[case::kseg0_exp1(0x9F00_0000)]
#[case::kseg1_exp1(0xBF00_0000)]
// scratchpad
#[case::kuseg_scratch(0x1F80_0000)]
#[case::kseg0_scratch(0x9F80_0000)]
// io ports
#[case::kuseg_io(0x1F80_1000)]
#[case::kseg0_io(0x9F80_1000)]
#[case::kseg1_io(0xBF80_1000)]
// expansion 2
#[case::kuseg_exp2(0x1F80_2000)]
#[case::kseg0_exp2(0x9F80_2000)]
#[case::kseg1_exp2(0xBF80_2000)]
// expansion 3
#[case::kuseg_exp3(0x1FA0_0000)]
#[case::kseg0_exp3(0x9FA0_0000)]
#[case::kseg1_exp3(0xBFA0_0000)]
// bios rom
// this will overwrite the bios lol
#[case::kuseg_bios(0x1FC0_0000)]
#[case::kseg0_bios(0x9FC0_0000)]
#[case::kseg1_bios(0xBFC0_0000)]
fn write_to_address(setup_tracing: (), #[case] base: u32) -> color_eyre::Result<()> {
    use pchan_emu::{cpu::ops::prelude::*, memory::PhysAddr};

    let mut emu = Emu::default();
    emu.cpu.pc = lookup_phys(KSEG0Addr(0xBFC0_0000).to_phys());
    emu.mem.write_array(
        KSEG1Addr(0xBFC0_0000),
        &[
            addiu(8, 0, 0),
            addiu(10, 0, 256),
            lui(9, (base >> 16) as i16),
            ori(9, 9, (base & 0x0000_FFFF) as i16),
            addu(11, 8, 9),
            sw(8, 11, 0),
            addiu(8, 8, 4),
            bne(8, 10, -16),
            nop(),
            OpCode(69420),
        ],
    );
    let summary = emu.step_jit_summarize::<JitSummary>()?;
    tracing::info!(?summary.function);

    assert_eq!(emu.cpu.gpr[9], base);
    assert_eq!(emu.cpu.gpr[8], 256);
    let base = lookup_phys(PhysAddr(base & 0x1FFF_FFFF));
    let slice = &emu.mem.as_ref()[(base as usize)..(base as usize + 256)];
    tracing::info!(?base);
    for i in 0..16 {
        let idx = base + i * 4;
        let idx = idx as usize;
        let value = &emu.mem.as_ref()[idx..(idx + 4)];
        let value = *value.as_array().unwrap();
        let value = u32::from_le_bytes(value);
        tracing::info!(?value);
        assert_eq!(value, i * 4);
    }

    tracing::info!(?summary.function);

    Ok(())
}
