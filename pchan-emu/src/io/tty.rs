use std::sync::Arc;

#[derive(derive_more::Debug, Clone)]
pub struct Tty {
    #[debug("buf: {}/1024", self.end)]
    buf:  Box<[u8]>,
    end:  usize,
    mode: TtyMode,
}

#[derive(derive_more::Debug, Clone)]
pub enum TtyMode {
    Stdout,
    Tracing,
    Channeled(flume::Sender<Arc<str>>),
    Silent,
}

impl Default for Tty {
    fn default() -> Self {
        Self {
            buf:  vec![0u8; 1024].into_boxed_slice(),
            end:  0,
            mode: TtyMode::Stdout,
        }
    }
}

impl Tty {
    pub fn putchar(&mut self, c: char) {
        if self.end == 1024 {
            tracing::error!("tty buffer overflow");
            return;
        }
        self.buf[self.end] = c as _;
        self.end += 1;
        if c == '\n' {
            _ = self.flush();
        }
    }

    pub fn flush(&mut self) -> color_eyre::Result<()> {
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

    pub fn set_channeled(&mut self) -> flume::Receiver<Arc<str>> {
        let (tx, rx) = flume::bounded(1024);
        self.mode = TtyMode::Channeled(tx);
        rx
    }

    pub fn set_tracing(&mut self) {
        self.mode = TtyMode::Tracing;
    }
}
