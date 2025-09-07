use std::backtrace::Backtrace;

use rstest::*;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self},
    prelude::*,
    util::SubscriberInitExt,
};

#[fixture]
pub fn setup_tracing() {
    _ = tracing_subscriber::registry()
        .with(
            fmt::layer().with_ansi(true), // .with_span_events(FmtSpan::CLOSE),
        )
        .with(EnvFilter::from_default_env())
        .try_init();
    std::panic::set_hook(Box::new(|info| {
        let (file, line, column) = info
            .location()
            .map(|loc| (loc.file(), loc.line(), loc.column()))
            .unwrap_or_default();
        tracing::error!(
            src.file = file,
            src.line = line,
            src.column = column,
            panic = %info.payload_as_str().unwrap_or_default()
        );
        let bt = Backtrace::capture();
        tracing::error!("backtrace: \n\n{}", bt);
        panic!();
    }));
}
