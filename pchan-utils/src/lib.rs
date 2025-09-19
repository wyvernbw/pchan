use std::backtrace::Backtrace;

use rstest::*;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self, format::FmtSpan},
    util::SubscriberInitExt,
};

use tracing_subscriber::{Layer, layer::SubscriberExt};

#[fixture]
pub fn setup_tracing() {
    _ = tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_ansi(true)
                .with_file(false)
                .with_line_number(false), // .with_span_events(FmtSpan::CLOSE),
        )
        // .with(
        //     fmt::layer()
        //         .with_ansi(true)
        //         .with_span_events(FmtSpan::CLOSE)
        //         .with_filter(
        //             EnvFilter::from_default_env()
        //                 // .add_directive("off".parse().unwrap())
        //                 .add_directive("pchan_emu[fn]=trace".parse().unwrap()),
        //         ),
        // )
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
