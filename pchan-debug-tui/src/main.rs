#![feature(associated_type_defaults)]
#![feature(iter_intersperse)]

use std::path::PathBuf;

use color_eyre::Result;
use serde::{Deserialize, Serialize};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tui_logger::TuiTracingSubscriberLayer;

#[path = "./app/app.rs"]
pub mod app;

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    bios_path: Option<PathBuf>,
}

impl AppConfig {
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
    tracing_subscriber::registry()
        .with(TuiTracingSubscriberLayer)
        .init();

    let app_config: AppConfig = confy::load("pchan-debugger", None)?;

    let terminal = ratatui::init();
    let result = app::run(app_config, terminal);
    ratatui::restore();
    result
}
