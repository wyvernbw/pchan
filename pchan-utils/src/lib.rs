#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]
#![feature(ptr_as_ref_unchecked)]

use std::{
    backtrace::Backtrace,
    sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use rstest::*;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{
    EnvFilter, Layer,
    fmt::{self, format::FmtSpan},
    util::SubscriberInitExt,
};

use tracing_subscriber::layer::SubscriberExt;

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

#[fixture]
pub fn setup_tracing() {
    let indicatif_layer = IndicatifLayer::new();
    _ = tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_ansi(true)
                .with_file(false)
                .without_time()
                .with_test_writer()
                .with_writer(indicatif_layer.get_stdout_writer())
                .with_line_number(false), // .with_span_events(FmtSpan::CLOSE),
        )
        .with(indicatif_layer)
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

#[macro_export]
macro_rules! array {
    ($($idx:literal => $val:expr),+ $(,)?) => (
        [$( $val ),+]
    );
}

use std::mem::size_of;

#[derive(derive_more::Deref, derive_more::Display)]
#[display("{}", self.0.as_str())]
pub struct Hex<const N: usize>(const_hex::Buffer<N, true>);

pub fn hex<T>(mut x: T) -> Hex<{ size_of::<T>() }> {
    let ptr = &mut x as *mut T as *mut u8;

    // SAFETY: should always be valid since size_of::<T> is enforced
    // at compile time
    let bytes = unsafe { &mut *ptr.cast::<[u8; size_of::<T>()]>() };

    if cfg!(target_endian = "little") {
        bytes.reverse();
    }

    Hex(const_hex::const_encode::<_, true>(bytes))
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

impl<'a, T> IgnorePoison<'a> for smol::lock::RwLock<T> {
    type Output = impl Future<Output = smol::lock::RwLockReadGuard<'a, T>>;

    type OutputMut = impl Future<Output = smol::lock::RwLockWriteGuard<'a, T>>;

    fn get(&'a self) -> Self::Output {
        self.read()
    }

    fn get_mut(&'a self) -> Self::OutputMut {
        self.write()
    }
}

pub fn default<T: Default>() -> T {
    T::default()
}
