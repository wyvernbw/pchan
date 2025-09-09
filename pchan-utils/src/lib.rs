use std::backtrace::Backtrace;

use rstest::*;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self, format::FmtSpan},
    prelude::*,
    util::SubscriberInitExt,
};

use tracing::{Level, info, info_span};
use tracing_subscriber::Registry;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, layer::SubscriberExt};

// Custom layer that prints span close events only for "my_span"
struct ExitPrinterLayer(String);

impl<S> Layer<S> for ExitPrinterLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_close(&self, id: tracing::span::Id, ctx: tracing_subscriber::layer::Context<S>) {
        if let Some(span) = ctx.span(&id)
            && span.name() == self.0
        {
            println!("Span '{}' closed!", span.name());
        }
    }
}

#[fixture]
pub fn setup_tracing() {
    _ = tracing_subscriber::registry()
        .with(
            fmt::layer().with_ansi(true), // .with_span_events(FmtSpan::CLOSE),
        )
        .with(
            fmt::layer()
                .with_ansi(true)
                .with_span_events(FmtSpan::CLOSE)
                .with_filter(
                    EnvFilter::from_default_env()
                        // .add_directive("off".parse().unwrap())
                        .add_directive("pchan_emu[fn]=trace".parse().unwrap()),
                ),
        )
        .with(
            EnvFilter::builder()
                .with_env_var("PCHAN_LOG")
                .from_env_lossy()
                .add_directive("cranelift_jit::backend=off".parse().unwrap()),
        )
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

#[macro_export]
macro_rules! array {
    ($($idx:literal => $val:expr),+ $(,)?) => (
        [$( $val ),+]
    );
}
