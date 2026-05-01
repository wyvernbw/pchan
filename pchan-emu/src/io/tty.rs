use std::{str::Utf8Error, sync::Arc};

use thiserror::Error;

use crate::memory::kb;

const TTY_CAP: usize = kb(16);

#[derive(derive_more::Debug, Clone)]
pub struct Tty {
    #[debug("buf: {}/{}", self.end, TTY_CAP)]
    buf:  Box<[u8]>,
    end:  usize,
    mode: TtyMode,
}

#[derive(derive_more::Debug, Clone)]
pub enum TtyMode {
    Stdout,
    Tracing,
    Channeled(kanal::Sender<Arc<str>>),
    Silent,
}

impl Default for Tty {
    fn default() -> Self {
        Self {
            buf:  vec![0u8; TTY_CAP].into_boxed_slice(),
            end:  0,
            mode: TtyMode::Stdout,
        }
    }
}

#[derive(Error, Debug)]
enum TtyFlushError {
    #[error("tty: invalid utf8: {0}")]
    Utf8Err(#[from] Utf8Error),
    #[error("tty: channel closed")]
    SendErr(#[from] kanal::SendError),
}

impl Tty {
    pub fn putchar(&mut self, c: char) {
        if self.end == TTY_CAP {
            tracing::error!("tty buffer overflow");
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
                print!("{}", string);
            }
            TtyMode::Tracing => {
                tracing::info!("{}", string.trim());
            }
            TtyMode::Channeled(tx) => {
                tx.send(string.to_owned().into())?;
            }
            TtyMode::Silent => {}
        }
        self.end = 0;
        Ok(())
    }

    pub fn set_channeled(&mut self) -> (kanal::Sender<Arc<str>>, kanal::Receiver<Arc<str>>) {
        let (tx, rx) = kanal::bounded(1024);
        self.mode = TtyMode::Channeled(tx.clone());
        (tx, rx)
    }

    pub fn set_channeled_with(&mut self, sender: kanal::Sender<Arc<str>>) {
        self.mode = TtyMode::Channeled(sender);
    }

    pub fn set_tracing(&mut self) {
        self.mode = TtyMode::Tracing;
    }
}
