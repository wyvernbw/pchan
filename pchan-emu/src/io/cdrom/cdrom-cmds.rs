use crate::cpu::Cpu;
use crate::io::cdrom::cdrom_format::{Bcd, Mss};
use crate::io::cdrom::{CDRomState, DriveStatus};
use bitbybit::*;
use pchan_utils::hex;
use smallvec::{SmallVec, smallvec};

use super::HInt;

#[derive(Debug, Clone)]
pub struct Response {
    pub int:  HInt,
    pub data: SmallVec<[u8; 8]>,
}

impl Response {
    pub fn new(int: HInt, data: SmallVec<[u8; 8]>) -> Self {
        Self { int, data }
    }
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

pub type ResponseList = SmallVec<[CdromResponse; 2]>;

impl CDRomState {
    fn drain_params(&mut self) -> impl Iterator<Item = u8> {
        std::iter::from_fn(|| self.param_fifo.pop_front())
    }
    pub fn send_cmd(&mut self, cmd: u8) -> ResponseList {
        self.status.set_busy_status(true);

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
                smallvec![CdromResponse::Immediate(self.int3_status())]
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
                        smallvec![CdromResponse::Immediate(self.int3_status())]
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
                match self.drive.drive_status {
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
                            CdromResponse::Immediate(self.int3_status()),
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
                let res1 = self.responses.insert(self.int3_status());
                let res2 = self.responses.insert(Response {
                    int:  HInt::Int2Complete,
                    data: smallvec![self.drive.status_code.raw_value()],
                });
                smallvec![
                    CdromResponse::InCycles(105, res1),
                    CdromResponse::InCycles(Cpu::CLOCK as u64, res2),
                ]
            }
            0x02 => self.setloc_cmd(),
            0x15 => self.seekl_cmd(),
            0x0e => self.setmode_cmd(),
            0x06 => self.readn_cmd(),
            cmd => {
                todo!("todo(cdrom): unhandled cmd: {}", hex(cmd));
            }
        }
    }

    fn int3_status(&self) -> Response {
        Response {
            int:  HInt::Int3Ack,
            data: smallvec![self.drive.status_code.raw_value()],
        }
    }
    fn int2_status(&self) -> Response {
        Response {
            int:  HInt::Int2Complete,
            data: smallvec![self.drive.status_code.raw_value()],
        }
    }

    fn get_param<T: From<u8>>(&mut self) -> T {
        self.param_fifo.pop_front().unwrap_or_default().into()
    }

    /// Setloc - Command 02h,amm,ass,asect --> INT3(stat)
    fn setloc_cmd(&mut self) -> ResponseList {
        self.status.set_busy_status(false);

        let min = self.get_param::<Bcd>();
        let sec = self.get_param::<Bcd>();
        let sect = self.get_param::<Bcd>();

        self.drive.setloc(Mss::new(min, sec, sect));
        let res = CdromResponse::Immediate(self.int3_status());
        smallvec![res]
    }

    /// SeekL - Command 15h --> INT3(stat) --> INT2(stat)
    fn seekl_cmd(&mut self) -> ResponseList {
        self.status.set_busy_status(false);
        let res1 = self.int3_status();
        let res2 = self.responses.insert(self.int2_status());
        smallvec![
            CdromResponse::Immediate(res1),
            CdromResponse::InCycles(100, res2)
        ]
    }

    /// Setmode - Command 0Eh,mode --> INT3(stat)
    fn setmode_cmd(&mut self) -> ResponseList {
        self.status.set_busy_status(false);
        let res = self.int3_status();
        let setmode = self.get_param::<SetMode>();
        self.drive.setmode(setmode);
        smallvec![CdromResponse::Immediate(res)]
    }

    /// ReadN - Command 06h --> INT3(stat) --> INT1(stat) --> datablock
    fn readn_cmd(&mut self) -> ResponseList {
        self.status.set_busy_status(false);
        self.drive.readn();
        smallvec![CdromResponse::Immediate(self.int3_status())]
    }
}

/// ```plaintext
///  7   Speed       (0=Normal speed, 1=Double speed)
///  6   XA-ADPCM    (0=Off, 1=Send XA-ADPCM sectors to SPU Audio Input)
///  5   Sector Size (0=800h=DataOnly, 1=924h=WholeSectorExceptSyncBytes)
///  4   Ignore Bit  (0=Normal, 1=Ignore Sector Size and Setloc position)
///  3   XA-Filter   (0=Off, 1=Process only XA-ADPCM sectors that match Setfilter)
///  2   Report      (0=Off, 1=Enable Report-Interrupts for Audio Play)
///  1   AutoPause   (0=Off, 1=Auto Pause upon End of Track) ;for Audio Play
///  0   CDDA        (0=Off, 1=Allow to Read CD-DA Sectors; ignore missing EDC)
/// ```
#[bitfield(u8, debug)]
#[derive(Default)]
pub struct SetMode {
    #[bit(0, rw)]
    cdda:       bool,
    #[bit(1, rw)]
    autopause:  bool,
    #[bit(2, rw)]
    report:     bool,
    #[bit(3, rw)]
    xa_filter:  bool,
    #[bit(4, rw)]
    ignore_bit: bool,
    #[bit(5, rw)]
    sect_size:  SetModeSectSize,
    #[bit(6, rw)]
    xa_adpcm:   bool,
    #[bit(7, rw)]
    speed:      SetModeSpeed,
}

impl const From<u8> for SetMode {
    fn from(value: u8) -> Self {
        SetMode::new_with_raw_value(value)
    }
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum SetModeSectSize {
    DataOnly0x800 = 0x0,
    Whole0x924    = 0x1,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum SetModeSpeed {
    Normal = 0x0,
    Double = 0x1,
}
