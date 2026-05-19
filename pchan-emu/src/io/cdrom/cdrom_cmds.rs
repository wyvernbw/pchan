use crate::io::cdrom::CDRomState;
use pchan_utils::hex;

use super::Int;

pub enum CdromIrqEvent {
    None,
    Immediate,
    InCycles(u64),
}

impl CDRomState {
    fn drain_params(&mut self) -> impl Iterator<Item = u8> {
        std::iter::from_fn(|| self.param_fifo.pop_front())
    }
    pub fn send_cmd(&mut self, cmd: u8) -> CdromIrqEvent {
        self.status.set_busy_status(true);

        match cmd {
            0x01 => {
                self.status.set_busy_status(false);
                CdromIrqEvent::Immediate
            }
            0x19 => {
                let Some(sub) = self.drain_params().next() else {
                    return CdromIrqEvent::None;
                };
                tracing::info!("cdrom: cmd 0x19");
                match sub {
                    // 20h INT3(yy,mm,dd,ver) Get cdrom BIOS date/version (yy,mm,dd,ver)
                    0x20 => {
                        self.status.set_busy_status(false);
                        for value in self.ver.iter() {
                            self.result_push(value);
                        }
                        CdromIrqEvent::Immediate
                    }

                    _ => {
                        tracing::warn!(
                            "todo(cdrom): cmd 0x19 (test) uhandled sub value: {}",
                            hex(sub)
                        );
                        CdromIrqEvent::None
                    }
                }
            }
            cmd => {
                tracing::warn!("todo(cdrom): unhandled cmd: {}", hex(cmd));
                CdromIrqEvent::None
            }
        }
    }
}
