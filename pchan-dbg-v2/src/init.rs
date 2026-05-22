use clap::Parser;
use miette::{Context, IntoDiagnostic, Result, miette};
use std::path::PathBuf;

pub struct EnvVars {
    pub bios_path: PathBuf,
    pub args:      CliArgs,
}

#[derive(clap::Parser)]
pub struct CliArgs {
    /// executable to run
    pub exe:    Option<PathBuf>,
    /// disc file to open
    ///
    /// supported formats: cue, bin, iso
    #[arg(long, short)]
    pub disc:   Option<PathBuf>,
    /// stream files. decreases ram usage, but increases load times
    ///
    /// default value: `true`
    #[arg(long, default_value_t = true)]
    pub stream: bool,
}

impl EnvVars {
    pub fn new() -> Result<Self> {
        let args = CliArgs::parse();
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
        Ok(Self { bios_path, args })
    }
}
