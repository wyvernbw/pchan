use pchan_emu::Emu;
use pchan_utils::setup_tracing;
use rstest::rstest;

#[rstest]
pub fn boot(setup_tracing: ()) -> color_eyre::Result<()> {
    let mut emu = Emu::default();
    emu.load_bios()?;
    emu.jump_to_bios();
    let result = emu.run();
    if let Err(err) = result {
        tracing::info!(?err);
    }
    Ok(())
}
