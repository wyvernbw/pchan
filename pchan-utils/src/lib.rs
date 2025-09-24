#![feature(ptr_as_ref_unchecked)]

use std::{backtrace::Backtrace, cell::Cell, fmt::Debug, time::Duration};

use rstest::*;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self},
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

use std::{mem::size_of, slice, str};

thread_local! {
    static BUFFERS: Buffers = const { Buffers::new() };
}

struct Buffers {
    slots: [[u8; 130]; 8], // 8 concurrent hex strings per thread
    index: Cell<usize>,
}

impl Buffers {
    const fn new() -> Self {
        Self {
            slots: [[0; 130]; 8],
            index: Cell::new(0),
        }
    }

    fn next_ptr(&self) -> *mut u8 {
        let i = self.index.get();
        self.index.set((i + 1) % self.slots.len());
        self.slots[i].as_ptr() as *mut u8
    }
}

pub fn hex<T>(mut x: T) -> &'static str {
    let ptr = &mut x as *mut T as *mut u8;
    let len = size_of::<T>();
    let bytes = unsafe { slice::from_raw_parts_mut::<u8>(ptr, len) };

    if cfg!(target_endian = "little") {
        bytes.reverse();
    }

    let bytes = bytes as &_;

    BUFFERS.with(|b| {
        let buf = b.next_ptr();
        let buf: &mut [u8; 130] = unsafe { &mut *(buf as *mut [u8; 130]) };
        let start = 2;
        let buf_pad = &mut buf[start..(start + len * 2)];

        let _ = const_hex::encode_to_slice(bytes, buf_pad);
        let buf = &mut buf[0..(start + len * 2)];
        buf[0] = b'0';
        buf[1] = b'x';
        unsafe { str::from_utf8_unchecked(buf) }
    })
}

#[cfg(test)]
#[test]
fn test_hex_encode() {
    let number = 0xDEAD_BEEFu32;
    let fmt = format!("0x{number:x}");
    let hex = hex(&number);
    assert_eq!(hex, fmt);
}
