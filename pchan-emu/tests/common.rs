use pchan_emu::Emu;
use rstest::fixture;

#[fixture]
pub fn emulator() -> Emu {
    Emu::default()
}
