use crate::io::cdrom::{CDRomState, DriveStatus};
use bitbybit::bitfield;
use pchan_utils::hex;

use super::HInt;

pub enum CdromIrqEvent {
    None,
    Immediate(HInt),
    InCycles(u64, HInt),
}

/// ```plaintext
///  7  Play          Playing CD-DA         ;\only ONE of these bits can be set
///  6  Seek          Seeking               ; at a time (ie. Read/Play won't get
///  5  Read          Reading data sectors  ;/set until after Seek completion)
///  4  ShellOpen     Once shell open (0=Closed, 1=Is/was Open)
///  3  IdError       (0=Okay, 1=GetID denied) (also set when Setmode.Bit4=1)
///  2  SeekError     (0=Okay, 1=Seek error)     (followed by Error Byte)
///  1  Spindle Motor (0=Motor off, or in spin-up phase, 1=Motor on)
///  0  Error         Invalid Command/parameters (followed by Error Byte)
/// ```
#[bitfield(u8, debug)]
pub struct StatusCode {
    #[bit(0, rw)]
    err:         bool,
    #[bit(1, rw)]
    spindle_mot: bool,
    #[bit(2, rw)]
    seek_err:    bool,
    #[bit(3, rw)]
    id_err:      bool,
    #[bit(4, rw)]
    shell_open:  bool,
    #[bit(5, rw)]
    read:        bool,
    #[bit(6, rw)]
    seek:        bool,
    #[bit(7, rw)]
    play:        bool,
}

impl Default for StatusCode {
    fn default() -> Self {
        Self::ZERO.with_spindle_mot(true)
    }
}

impl CDRomState {
    fn drain_params(&mut self) -> impl Iterator<Item = u8> {
        std::iter::from_fn(|| self.param_fifo.pop_front())
    }
    pub fn send_cmd(&mut self, cmd: u8) -> [CdromIrqEvent; 2] {
        fn one(event: CdromIrqEvent) -> [CdromIrqEvent; 2] {
            [event, CdromIrqEvent::None]
        }

        self.status.set_busy_status(true);

        match cmd {
            0x01 => {
                tracing::info!("0x01 nop");
                self.status.set_busy_status(false);
                self.result_push(self.status_code.raw_value());
                one(CdromIrqEvent::Immediate(HInt::Int3Ack))
            }
            0x19 => {
                tracing::info!("0x19 test command");
                let Some(sub) = self.drain_params().next() else {
                    return one(CdromIrqEvent::None);
                };
                tracing::info!("cdrom: cmd 0x19");
                match sub {
                    // 20h INT3(yy,mm,dd,ver) Get cdrom BIOS date/version (yy,mm,dd,ver)
                    0x20 => {
                        self.status.set_busy_status(false);
                        for value in self.ver.iter() {
                            self.result_push(value);
                        }
                        one(CdromIrqEvent::Immediate(HInt::Int3Ack))
                    }

                    _ => {
                        tracing::warn!(
                            "todo(cdrom): cmd 0x19 (test) uhandled sub value: {}",
                            hex(sub)
                        );
                        one(CdromIrqEvent::None)
                    }
                }
            }
            // 0x1a INT3(stat) --> INT2/5 (stat,flags,type,atip,"SCEx")
            0x1a => {
                tracing::info!("0x1a INT3(stat) -> INT2/5(...)");
                match self.drive_status {
                    DriveStatus::LidOpen => {
                        self.result_push(0x11);
                        self.result_push(0x80);
                        self.status.set_busy_status(false);
                        one(CdromIrqEvent::Immediate(HInt::Int5DiskErr))
                    }
                    DriveStatus::SpinUp => {
                        self.result_push(0x01);
                        self.result_push(0x80);
                        self.status.set_busy_status(false);
                        one(CdromIrqEvent::Immediate(HInt::Int5DiskErr))
                    }
                    DriveStatus::DetectBusy => {
                        self.result_push(0x03);
                        self.result_push(0x80);
                        self.status.set_busy_status(false);
                        one(CdromIrqEvent::Immediate(HInt::Int5DiskErr))
                    }
                    DriveStatus::NoDisk => {
                        self.result_push(self.status_code.raw_value());
                        self.result_push(0x08);
                        self.result_push(0x40);
                        [
                            CdromIrqEvent::Immediate(HInt::Int3Ack),
                            CdromIrqEvent::Immediate(HInt::Int5DiskErr),
                        ]
                    }
                    DriveStatus::AudioDisk => todo!(),
                    DriveStatus::LicensedMode2 => todo!(),
                }
            }
            cmd => {
                tracing::warn!("todo(cdrom): unhandled cmd: {}", hex(cmd));
                one(CdromIrqEvent::None)
            }
        }
    }
}
