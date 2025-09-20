#![feature(ptr_as_ref_unchecked)]
use std::{backtrace::Backtrace, cell::Cell};

use rstest::*;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self},
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
                .without_time()
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

use std::{mem::size_of, slice, str};

thread_local! {
    static BUFFERS: Buffers = const { Buffers::new() };
}

struct Buffers {
    slots: [[u8; 130]; 4], // 4 concurrent hex strings per thread
    index: Cell<usize>,
}

impl Buffers {
    const fn new() -> Self {
        Self {
            slots: [[0; 130]; 4],
            index: Cell::new(0),
        }
    }

    fn next_ptr(&self) -> *mut u8 {
        let i = self.index.get();
        self.index.set((i + 1) % self.slots.len());
        self.slots[i].as_ptr() as *mut u8
    }
}

pub fn hex<T>(x: &T) -> &'static str {
    let ptr = x as *const T as *const u8;
    let len = size_of::<T>();
    let bytes = unsafe { slice::from_raw_parts(ptr, len) };

    BUFFERS.with(|b| {
        let buf = b.next_ptr();
        let buf: &mut [u8; 130] = unsafe { &mut *(buf as *mut [u8; 130]) };
        let start = 2;
        let buf_pad = &mut buf[start..(start + len * 2)];

        let _ = const_hex::encode_to_slice(bytes, buf_pad);
        if cfg!(target_endian = "little") {
            let byte_len = len * 2;
            for i in 0..(len / 2) {
                let byte_idx = i * 2;
                buf_pad.swap(byte_idx, byte_len - byte_idx - 2);
                buf_pad.swap(byte_idx + 1, byte_len - byte_idx - 2 + 1);
            }
        }
        let buf = &mut buf[0..(start + len * 2)];
        buf[0] = b'0';
        buf[1] = b'x';
        unsafe { str::from_utf8_unchecked(buf) }
    })
}
