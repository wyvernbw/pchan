#[cfg(test)]
#[test]
fn run() -> color_eyre::Result<()> {
    use pchan_emu::{Emu, bootloader::Bootloader, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.load_bios()?;
    emu.cpu.jump_to_bios();

    loop {
        use pchan_utils::hex;

        PipelineV2::new(&emu).run_once(&mut emu)?;
        // tracing::info!(r10 = hex(emu.cpu.gpr[10]), r11 = hex(emu.cpu.gpr[11]));
        // if emu.cpu.gpr[10] == 0x00000f80 {
        //     tracing::info!(?emu);
        // }
        // if emu.cpu.gpr[11] == 0x00000f80 {
        //     assert!(emu.cpu.gpr[10] <= emu.cpu.gpr[11]);
        // };
        tracing::info!(pc = hex(emu.cpu.pc));
        match inquire::prompt_confirmation("continue?") {
            Ok(true) => {}
            _ => break,
        }
    }

    Ok(())
}
