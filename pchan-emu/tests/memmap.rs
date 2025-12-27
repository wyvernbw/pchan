#![feature(portable_simd)]
#![allow(unused_variables)]

use pchan_emu::cpu::ops::OpCode;
use pchan_emu::cpu::ops::*;
use pchan_emu::dynarec_v2::PipelineV2;
use pchan_emu::memory::fastmem::LUT;
use pchan_emu::memory::{MEM_MAP, ext};
use pchan_emu::{Emu, memory};
use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;

#[rstest]
// ram
#[case::kuseg_ram(0x0000_0000 + 44)]
#[case::kseg0_ram(0x8000_0000 + 44)]
#[case::kseg1_ram(0xA000_0000 + 44)]
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
fn write_to_address(setup_tracing: (), #[case] base: u32) -> color_eyre::Result<()> {
    use pchan_emu::{cpu::program, io::IO};

    let mut emu = Emu::default();
    let count = memory::kb(8) / 4;
    emu.cpu.pc = 0x0000_0000;
    let main = program([
        addiu(8, 0, 0),
        addiu(10, 0, count as _),
        lui(9, (base >> 16) as i16),
        ori(9, 9, (base & 0x0000_FFFF) as i16),
        addu(11, 8, 9),
        sw(8, 11, 0),
        addiu(8, 8, 4),
        bne(8, 10, -4),
        nop(),
        OpCode::HALT,
    ]);
    emu.write_many::<u32>(0x0000_0000, &main);

    PipelineV2::new(&emu).run_once(&mut emu)?;
    PipelineV2::new(&emu).run_once(&mut emu)?;

    assert_eq!(emu.cpu.gpr[9], base);
    assert_eq!(emu.cpu.gpr[8], count as _);

    tracing::info!(?base);
    let mut address = base;
    for i in 0..(count as u32 / 4u32) {
        let value = emu.read::<u32>(address);
        tracing::info!(?value);
        assert_eq!(value, i * 4);
        address += 4;
    }

    emu.write(0x80008670, 0x0000_1234u32);
    let written = emu
        .mem
        .read_region::<u32>(MEM_MAP.ram, MEM_MAP.ram, 0x80008670);
    tracing::info!(?written);
    tracing::info!(lut_read = ?&LUT.read[0..4]);
    tracing::info!(lut_write = ?&LUT.write[0..4]);
    assert_eq!(
        written, 0x00001234,
        "read region failed. write is most likely broken"
    );
    assert_eq!(
        emu.read::<u32>(0x80008670),
        0x00001234u32,
        "virtual read failed. read is most likely broken."
    );

    Ok(())
}
