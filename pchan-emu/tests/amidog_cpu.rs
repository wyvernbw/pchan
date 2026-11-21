#[cfg(test)]
#[test]
fn run() -> color_eyre::Result<()> {
    use std::alloc::Global;

    use pchan_emu::Emu;
    use pchan_emu::run::Runner;
    use pchan_utils::setup_tracing;

    if !cfg!(feature = "amidog-tests") {
        return Ok(());
    }

    setup_tracing();
    let mut emu = Emu::default();
    let mut runner = Runner::new();
    emu.load_bios()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();

    loop {
        runner.execute(&mut emu);
    }
}
