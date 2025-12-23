#![deny(clippy::unwrap_used)]
#![allow(clippy::collapsible_if)]
#![feature(try_blocks)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_collect_into)]
#![feature(associated_type_defaults)]
#![feature(iter_intersperse)]

use std::{backtrace::Backtrace, io::stdout, panic::catch_unwind, path::PathBuf};

use color_eyre::{Result, eyre::eyre};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

#[path = "./app/app.rs"]
pub mod app;
pub mod utils;

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    bios_path: Option<PathBuf>,
}

impl AppConfig {
    #[must_use]
    pub fn initialized(&self) -> bool {
        match self {
            AppConfig { bios_path: None } => false,
            AppConfig { bios_path: _ } => true,
        }
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tui_logger::init_logger(tui_logger::LevelFilter::Trace)?;
    let logs = confy::get_configuration_file_path("pchan-debugger", Some("logs.txt"))?;
    let logs = std::fs::File::create(logs)?;

    // std::panic::set_hook(Box::new(|info| {
    //     let (file, line, column) = info
    //         .location()
    //         .map(|loc| (loc.file(), loc.line(), loc.column()))
    //         .unwrap_or_default();
    //     tracing::error!(src.file = file,src.line = line,src.column = column,panic =  %info.payload_as_str().unwrap_or_default());
    //     let bt = Backtrace::capture();
    //     tracing::error!("backtrace: \n\n{}", bt);
    //     ratatui::restore();
    // }));

    tracing_subscriber::fmt()
        .with_writer(logs)
        .with_env_filter(EnvFilter::from_env("PCHAN_LOG"))
        .with_ansi(false)
        .with_filter_reloading()
        .init();
    // tracing_subscriber::registry()
    //     .with(TuiTracingSubscriberLayer)
    //     .init();

    let app_config: AppConfig = confy::load("pchan-debugger", Some("config"))?;

    let terminal = ratatui::init();
    let result = app::run(app_config, terminal);
    ratatui::restore();
    result
}
