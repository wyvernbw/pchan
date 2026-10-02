#![feature(const_destruct)]
#![feature(const_trait_impl)]
#![allow(incomplete_features)]

use std::fmt::Display;
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use kanal::{AsyncReceiver, AsyncSender, Receiver, Sender};

#[cfg(feature = "tracing-subscriber")]
pub mod trace_utils {
    use std::backtrace::Backtrace;

    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::fmt::{self};
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Layer};

    use tracing_subscriber::layer::SubscriberExt;

    #[cfg_attr(test, rstest::fixture)]
    pub fn setup_tracing() {
        _ = tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .with_ansi(true)
                    .with_file(false)
                    .without_time()
                    .with_test_writer()
                    .with_line_number(false), // .with_span_events(FmtSpan::CLOSE),
            )
            .with(
                fmt::layer()
                    .with_ansi(false)
                    .with_file(false)
                    .without_time()
                    .with_writer(std::fs::File::create("pchan.log").unwrap())
                    .with_line_number(false), // .with_span_events(FmtSpan::CLOSE),
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
                    .with_default_directive("info".parse().unwrap())
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
        }));
    }

    pub struct InitTracingArgs {
        pub stdout:     bool,
        pub file:       bool,
        pub panic_hook: bool,
    }

    impl Default for InitTracingArgs {
        fn default() -> Self {
            Self {
                stdout:     true,
                file:       true,
                panic_hook: true,
            }
        }
    }

    pub fn init_tracing(
        InitTracingArgs {
            stdout,
            file,
            panic_hook,
        }: InitTracingArgs,
    ) {
        let stdout_layer = stdout.then(|| {
            fmt::layer()
                .with_ansi(true)
                .with_file(false)
                .without_time()
                .with_test_writer()
                .with_line_number(false)
                .with_writer(std::io::stdout)
        });

        let file_layer = file.then(|| {
            fmt::layer()
                .with_ansi(false)
                .with_file(false)
                .without_time()
                .with_writer(std::fs::File::create("pchan.log").unwrap())
                .with_line_number(false)
        });

        let span_layer = fmt::layer()
            .with_ansi(true)
            .with_span_events(FmtSpan::CLOSE)
            .with_filter(
                EnvFilter::from_default_env().add_directive("pchan_emu[fn]=trace".parse().unwrap()),
            );

        let env_filter = EnvFilter::builder()
            .with_env_var("PCHAN_LOG")
            .with_default_directive("info".parse().unwrap())
            .from_env_lossy();

        _ = tracing_subscriber::registry()
            .with(stdout_layer)
            .with(file_layer)
            .with(span_layer)
            .with(env_filter)
            .try_init();

        if panic_hook {
            let old_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                old_hook(info);
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
            }));
        }
    }
}

#[cfg(feature = "tracy")]
pub mod tracy {
    #[derive(derive_more::Deref, derive_more::DerefMut, Clone, derive_more::Debug)]
    #[debug("N/A")]
    pub struct TracyClient(tracy_client::Client);

    impl Default for TracyClient {
        fn default() -> Self {
            Self(tracy_client::Client::start())
        }
    }

    pub use tracy_client::*;
}

#[cfg(feature = "tracing-subscriber")]
pub use trace_utils::*;

pub fn default_const<T: Default>() -> T {
    T::default()
}

pub const fn max_simd_width_bytes() -> usize {
    if cfg!(target_feature = "avx512f") {
        return 64;
    } // 512 bits

    if cfg!(target_feature = "neon") {
        return 16;
    }

    if cfg!(target_feature = "avx2") {
        return 32;
    } // 256 bits

    if cfg!(target_feature = "sse2") {
        return 16;
    } // 128 bits

    1
}

pub const MAX_SIMD_WIDTH: usize = max_simd_width_bytes();

pub type Chan<T> = (Sender<T>, Receiver<T>);
pub type AsyncChan<T> = (AsyncSender<T>, AsyncReceiver<T>);

#[macro_export]
macro_rules! array {
    ($($idx:literal => $val:expr),+ $(,)?) => (
        [$( $val ),+]
    );
}

use std::mem::size_of;

const PTR_SIZE: usize = size_of::<usize>();

pub struct Hex<const PREFIX: bool> {
    buf: [u8; PTR_SIZE * 2],
    len: usize,
}

pub fn hex<T>(x: T) -> Hex<true> {
    hex_pref::<T, true>(x)
}

pub fn hex_pref<T, const PREFIX: bool>(mut x: T) -> Hex<PREFIX> {
    assert_hex_size::<T>();

    let ptr = &mut x as *mut T as *mut u8;

    // SAFETY: should always be valid since size_of::<T> is enforced
    // at compile time
    let value = unsafe { core::slice::from_raw_parts_mut(ptr, size_of_val(&x)) };

    if cfg!(target_endian = "little") {
        value.reverse();
    }
    let mut bytes = [0u8; PTR_SIZE];
    bytes[..size_of::<T>()].copy_from_slice(value);

    let mut sink = [b'0'; PTR_SIZE * 2];
    // should not error
    let _ = const_hex::encode_to_slice(bytes, &mut sink).expect("whatt");
    Hex {
        buf: sink,
        len: size_of::<T>() * 2,
    }
}

impl<const PREFIX: bool> Display for Hex<PREFIX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = unsafe { str::from_utf8_unchecked(&self.buf[..self.len]) };
        match PREFIX {
            true => {
                write!(f, "0x{str}")
            }
            false => write!(f, "{str}"),
        }
    }
}

const fn assert_hex_size<T>() {
    assert!(
        size_of::<T>() <= PTR_SIZE,
        "value passed to hex function is bigger than the pointer size."
    )
}

#[cfg(test)]
#[test]
fn test_hex_encode() {
    let number = 0xDEAD_BEEFu32;
    let fmt = format!("0x{number:x}");
    let hex = hex(number);
    assert_eq!(hex.to_string(), fmt);
}

pub trait IgnorePoison<'a> {
    type Output;
    type OutputMut;

    fn get(&'a self) -> Self::Output;
    fn get_mut(&'a self) -> Self::OutputMut;
}

impl<'a, T> IgnorePoison<'a> for Mutex<T>
where
    T: 'a,
{
    type Output = MutexGuard<'a, T>;
    type OutputMut = MutexGuard<'a, T>;

    fn get(&'a self) -> Self::Output {
        self.lock().unwrap()
    }

    fn get_mut(&'a self) -> Self::OutputMut {
        self.lock().unwrap()
    }
}

impl<'a, T> IgnorePoison<'a> for RwLock<T>
where
    T: 'a,
{
    type Output = RwLockReadGuard<'a, T>;
    type OutputMut = RwLockWriteGuard<'a, T>;

    fn get(&'a self) -> Self::Output {
        self.read().unwrap()
    }

    fn get_mut(&'a self) -> Self::OutputMut {
        self.write().unwrap()
    }
}

pub fn default<T: Default>() -> T {
    T::default()
}
