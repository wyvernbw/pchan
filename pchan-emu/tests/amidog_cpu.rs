#[cfg(test)]
#[test]
fn run() -> color_eyre::Result<()> {
    use pchan_emu::{Emu, bootloader::Bootloader, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.load_bios()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();

    if !cfg!(feature = "amidog-tests") {
        panic!("amidog tests feature is not enabled.")
    }

    loop {
        PipelineV2::new(&emu).run_once(&mut emu)?;
    }
}
