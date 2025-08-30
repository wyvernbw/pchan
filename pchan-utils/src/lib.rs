use rstest::*;
use tracing_subscriber::{EnvFilter, fmt, prelude::*, util::SubscriberInitExt};

#[fixture]
pub fn setup_tracing() {
    _ = tracing_subscriber::registry()
        .with(fmt::layer().with_ansi(true))
        .with(EnvFilter::from_default_env())
        .try_init();
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().map(|loc| loc.file()).unwrap_or_default();
        tracing::error!(
            src = location,
            panic = %info.payload_as_str().unwrap_or_default()
        );
    }));
}
