use crate::cpu::Cpu;
use crate::io::cdrom::{CDRomState, DriveStatus};
use bitbybit::bitfield;
use pchan_utils::hex;
use smallvec::{SmallVec, smallvec};

use super::HInt;

#[derive(Debug, Clone)]
pub struct Response {
    pub int:  HInt,
    pub data: SmallVec<[u8; 8]>,
}

pub type ResponseId = usize;

#[derive(Debug, Clone)]
pub enum CdromResponse {
    None,
    Immediate(Response),
    InCycles(u64, ResponseId),
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
        Self::ZERO
    }
}

impl CDRomState {
    fn drain_params(&mut self) -> impl Iterator<Item = u8> {
        std::iter::from_fn(|| self.param_fifo.pop_front())
    }
    pub fn send_cmd(&mut self, cmd: u8) -> SmallVec<[CdromResponse; 2]> {
        self.status.set_busy_status(true);
        let arena = &mut self.responses;

        fn diskerr(data: &[u8]) -> SmallVec<[CdromResponse; 2]> {
            smallvec![CdromResponse::Immediate(Response {
                int:  HInt::Int5DiskErr,
                data: SmallVec::from_slice(data),
            })]
        }

        match cmd {
            0x01 => {
                tracing::info!("0x01 nop");
                self.status.set_busy_status(false);
                smallvec![CdromResponse::Immediate(Response {
                    int:  HInt::Int3Ack,
                    data: smallvec![self.status_code.raw_value()],
                })]
            }
            0x19 => {
                tracing::info!("0x19 test command");
                let Some(sub) = self.drain_params().next() else {
                    return smallvec![CdromResponse::None];
                };
                tracing::info!("cdrom: cmd 0x19");
                match sub {
                    // 20h INT3(yy,mm,dd,ver) Get cdrom BIOS date/version (yy,mm,dd,ver)
                    0x20 => {
                        self.status.set_busy_status(false);
                        smallvec![CdromResponse::Immediate(Response {
                            int:  HInt::Int3Ack,
                            data: SmallVec::from_slice(self.ver.as_slice()),
                        })]
                    }

                    _ => {
                        tracing::warn!(
                            "todo(cdrom): cmd 0x19 (test) uhandled sub value: {}",
                            hex(sub)
                        );
                        smallvec![CdromResponse::None]
                    }
                }
            }
            // 0x1a INT3(stat) --> INT2/5 (stat,flags,type,atip,"SCEx")
            0x1a => {
                tracing::info!("0x1a INT3(stat) -> INT2/5(...)");
                match self.drive_status {
                    DriveStatus::LidOpen => {
                        self.status.set_busy_status(false);
                        diskerr(&[0x11, 0x80])
                    }
                    DriveStatus::SpinUp => {
                        self.status.set_busy_status(false);
                        diskerr(&[0x01, 0x80])
                    }
                    DriveStatus::DetectBusy => {
                        self.status.set_busy_status(false);
                        diskerr(&[0x03, 0x80])
                    }
                    DriveStatus::NoDisk => {
                        self.status.set_busy_status(false);
                        smallvec![
                            CdromResponse::Immediate(Response {
                                int:  HInt::Int3Ack,
                                data: smallvec![0x08, 0x40],
                            }),
                            CdromResponse::Immediate(Response {
                                int:  HInt::Int5DiskErr,
                                data: smallvec![],
                            }),
                        ]
                    }
                    DriveStatus::AudioDisk => todo!(),
                    // INT3(stat), INT2(02h,00h, 20h,00h, 53h,43h,45h,4xh)
                    DriveStatus::LicensedMode2 => {
                        self.status.set_busy_status(false);
                        smallvec![
                            CdromResponse::Immediate(Response {
                                int:  HInt::Int3Ack,
                                data: smallvec![self.status_code.raw_value()],
                            }),
                            CdromResponse::Immediate(Response {
                                int:  HInt::Int2Complete,
                                data: smallvec![0x02, 0x00, 0x20, 0x00, 0x53, 0x43, 0x45, 0x49],
                            }),
                        ]
                    }
                }
            }
            // ReadTOC - Command 1Eh --> INT3(stat) --> INT2(stat)
            0x1e => {
                self.status.set_busy_status(false);
                tracing::info!("ReadTOC INT3(stat) --> INT2(stat)");
                let res1 = arena.insert(Response {
                    int:  HInt::Int3Ack,
                    data: smallvec![self.status_code.raw_value()],
                });
                let res2 = arena.insert(Response {
                    int:  HInt::Int2Complete,
                    data: smallvec![self.status_code.raw_value()],
                });
                smallvec![
                    CdromResponse::InCycles(105, res1),
                    CdromResponse::InCycles(Cpu::CLOCK as u64, res2),
                ]
            }
            cmd => {
                todo!("todo(cdrom): unhandled cmd: {}", hex(cmd));
            }
        }
    }
}
