#![deny(clippy::unwrap_used)]
#![allow(clippy::collapsible_if)]
#![feature(try_blocks)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_collect_into)]
#![feature(associated_type_defaults)]
#![feature(iter_intersperse)]

use std::path::PathBuf;

use color_eyre::Result;
use serde::{Deserialize, Serialize};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};
use tui_logger::TuiTracingSubscriberLayer;

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
