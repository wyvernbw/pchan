#![feature(portable_simd)]
#![allow(unused_variables)]

use pchan_emu::Emu;
use pchan_emu::dynarec::prelude::*;
use pchan_emu::jit::JIT;
use pchan_emu::memory::ext;
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
    let mut emu = Emu::default();
    let mut jit = JIT::default();
    emu.cpu.pc = 0xBFC0_0000;
    let main = program([
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
    ]);
    emu.write_many::<u32>(0xBFC0_0000, &main);
    let summary = emu.step_jit_summarize::<JitSummary>(&mut jit)?;
    tracing::info!(?summary.function);

    assert_eq!(emu.cpu.gpr[9], base);
    assert_eq!(emu.cpu.gpr[8], 256);

    tracing::info!(?base);
    let mut address = base;
    for i in 0..16 {
        let value = emu.read::<u32, ext::NoExt>(address);
        tracing::info!(?value);
        assert_eq!(value, i * 4);
        address += 4;
    }

    tracing::info!(?summary.function);

    Ok(())
}
