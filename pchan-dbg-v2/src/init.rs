use miette::{Context, IntoDiagnostic, Result, miette};
use std::path::PathBuf;

pub struct EnvVars {
    pub bios_path: PathBuf,
    pub exe_path:  Option<PathBuf>,
}

impl EnvVars {
    pub fn new() -> Result<Self> {
        let bios_path: PathBuf = std::env::var("PCHAN_BIOS")
            .map_err(|err| {
                miette!(
                    code = "bios::unset",
                    help = "try setting the `PCHAN_BIOS` env variable.",
                    "error finding bios file!",
                )
                .wrap_err(err)
            })?
            .parse()
            .into_diagnostic()
            .wrap_err("value in PCHAN_BIOS is not a valid path.")?;
        let exe_path = std::env::args()
            .skip(1)
            .next()
            .map(|exe_path| PathBuf::from(exe_path));
        Ok(Self {
            bios_path,
            exe_path,
        })
    }
}
