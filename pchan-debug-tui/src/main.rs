use color_eyre::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tui_logger::TuiTracingSubscriberLayer;

pub mod app;

fn main() -> Result<()> {
    color_eyre::install()?;
    tui_logger::init_logger(tui_logger::LevelFilter::Trace)?;
    tracing_subscriber::registry()
        .with(TuiTracingSubscriberLayer)
        .init();
    let terminal = ratatui::init();
    let result = app::run(terminal);
    ratatui::restore();
    result
}
