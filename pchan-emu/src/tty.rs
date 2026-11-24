use core::str::Utf8Error;

use thiserror::Error;

use crate::memory::kb;

const TTY_CAP: usize = kb(16);

#[derive(derive_more::Debug, Clone)]
pub struct Tty {
    #[debug("buf: {}/{}", self.end, TTY_CAP)]
    pub buf: Box<[u8]>,
    end:     usize,
    mode:    TtyMode,
}

#[derive(derive_more::Debug, Clone)]
pub enum TtyMode {
    Stdout,
    Tracing,
    Silent,
}

impl Tty {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf:  unsafe { Box::new_zeroed_slice(TTY_CAP).assume_init() },
            end:  0,
            mode: TtyMode::Stdout,
        }
    }
}

impl Default for Tty {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Error, Debug)]
pub enum TtyFlushError {
    #[error("tty: invalid utf8: {0}")]
    Utf8Err(#[from] Utf8Error),
    #[error("tty: channel closed")]
    SendErr(#[from] kanal::SendError),
}

impl Tty {
    pub fn putchar(&mut self, c: char) {
        if self.end == TTY_CAP {
            tracing::error!(target: "tty", "(p-chan) tty buffer overflow");
            return;
        }
        self.buf[self.end] = c as _;
        self.end += 1;
        if c == '\n' {
            _ = self.flush();
        }
    }

    pub fn flush(&mut self) -> Result<(), TtyFlushError> {
        let string = str::from_utf8(&self.buf.as_ref()[..self.end])?;
        match &mut self.mode {
            TtyMode::Stdout => {
                print!("{string}");
            }
            TtyMode::Tracing => {
                tracing::info!(target: "tty", "{}", string.trim());
            }
            TtyMode::Silent => {}
        }
        self.end = 0;
        Ok(())
    }

    pub fn set_tracing(&mut self) {
        self.mode = TtyMode::Tracing;
    }
}
