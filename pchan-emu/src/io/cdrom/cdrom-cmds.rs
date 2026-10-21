use core::alloc::Allocator;
use core::iter;

use crate::Emu;
use crate::cpu::{Cpu, cycles_to_ms};
use crate::io::cdrom::cdrom_format::{Bcd, CdromCursor, Mss};
use crate::io::cdrom::{CDRomState, DriveStatus};
use crate::io::evque::{EvCtx, EventId, EventKind};
use bitbybit::{bitenum, bitfield};
use pchan_encode::derive::Endecode;
use pchan_utils::hex;

use super::HInt;

#[derive(Debug, Clone)]
pub struct ResponseV2 {
    pub int:  HInt,
    pub data: heapless::Vec<u8, 8>,
}

impl ResponseV2 {
    pub fn new(int: HInt, data: impl Into<heapless::Vec<u8, 8>>) -> Self {
        Self {
            int,
            data: data.into(),
        }
    }
}

fn int1(data: impl Into<heapless::Vec<u8, 8>>) -> ResponseV2 {
    ResponseV2::new(HInt::Int1DataReady, data.into())
}
fn int2(data: impl Into<heapless::Vec<u8, 8>>) -> ResponseV2 {
    ResponseV2::new(HInt::Int2Complete, data.into())
}
fn int3(data: impl Into<heapless::Vec<u8, 8>>) -> ResponseV2 {
    ResponseV2::new(HInt::Int3Ack, data.into())
}
fn int4(data: impl Into<heapless::Vec<u8, 8>>) -> ResponseV2 {
    ResponseV2::new(HInt::Int4DataEnd, data.into())
}
fn int5(data: impl Into<heapless::Vec<u8, 8>>) -> ResponseV2 {
    ResponseV2::new(HInt::Int5DiskErr, data.into())
}

impl<const N: usize> From<StatusCode> for heapless::Vec<u8, N> {
    fn from(value: StatusCode) -> Self {
        Self::from_array([value.raw_value()])
    }
}

#[derive(Default, derive_more::Debug, Endecode)]
pub(crate) enum CommandStateV2 {
    #[default]
    Idle,
    Nop,
    GetIdInt3_5,
    GetIdInt2_5,

    TestHC05BiosDateInt3,

    ReadTOCInt3,
    ReadTOCInt2,

    SetlocInt3(Mss<Bcd>),

    SeekLInt3,
    SeekLInt2,

    SetModeInt3(SetMode),

    PauseInt3,
    PauseInt2,

    ReadNInt3,
    ReadNInt1,

    DemuteInt3,

    InitInt3,
    InitInt2,

    GetTNInt3,

    GetTDInt3(Bcd),
}

const NOP_TIMING: u32 = 0x000c4e1;
const GETID_RES_2_TIMING: u32 = 0x0004a00;
const INIT_TIMING: u32 = 0x0013cce;
const PAUSE_2X_TIMING: u32 = 0x010bd93;
const PAUSE_1X_TIMING: u32 = 0x021181c;

impl<A: Allocator> Emu<A> {
    pub(super) fn cdrom_send_cmd_v2(&mut self, cmd: u8) {
        self.cdrom.status.set_busy_status(true);
        match cmd {
            // nop
            0x01 => {
                self.cdrom_continue_cmd(CommandStateV2::Nop);
                self.cdrom_queue_response(NOP_TIMING);
            }
            // Setloc - Command 02h,amm,ass,asect --> INT3(stat)
            0x02 => {
                let min = self.cdrom.get_param::<Bcd>();
                let sec = self.cdrom.get_param::<Bcd>();
                let sect = self.cdrom.get_param::<Bcd>();
                let mss = Mss::new(min, sec, sect);
                tracing::info!(target: "cdrom", "Setloc\t\t{}->{}", self.cdrom.drive.cursor.to_mss::<u8>(), mss);
                self.cdrom_continue_cmd(CommandStateV2::SetlocInt3(mss));
                self.cdrom_queue_response(NOP_TIMING);
            }
            // `ReadN` - Command 06h --> INT3(stat) --> INT1(stat) --> datablock
            0x06 => {
                let seek_time = self.cdrom.drive.current_seek_time();
                tracing::info!(target: "cdrom", "ReadN\t\t{} seek time: {}ms", self.cdrom.drive.cursor.to_mss::<u8>(), cycles_to_ms(seek_time));
                self.cdrom_continue_cmd(CommandStateV2::ReadNInt3);
                self.cdrom_queue_response(NOP_TIMING);
            }
            // Demute - Command 0Ch --> INT3(stat)
            0x0c => {
                self.cdrom_continue_cmd(CommandStateV2::DemuteInt3);
                self.cdrom_queue_response(NOP_TIMING);
            }
            // Setmode - Command 0Eh,mode --> INT3(stat)
            0x0e => {
                tracing::info!(target: "cdrom", "Setmode\t\t{}", self.cdrom.drive.cursor.to_mss::<u8>());
                let setmode = self.cdrom.get_param::<SetMode>();
                self.cdrom_continue_cmd(CommandStateV2::SetModeInt3(setmode));
                self.cdrom_queue_response(NOP_TIMING);
            }
            // Pause - Command 09h --> INT3(stat) --> INT2(stat)
            0x09 => {
                tracing::info!(target: "cdrom", "Pause\t\t{}", self.cdrom.drive.cursor.to_mss::<u8>());
                self.cdrom_continue_cmd(CommandStateV2::PauseInt3);
                self.cdrom_queue_response(NOP_TIMING);
            }
            // Init - Command 0Ah --> INT3(stat) --> INT2(stat)
            0x0a => {
                self.cdrom_continue_cmd(CommandStateV2::InitInt3);
                self.cdrom_queue_response(INIT_TIMING);
            }
            // GetTN - Command 13h --> INT3(stat,first,last) ;BCD
            0x13 => {
                self.cdrom_continue_cmd(CommandStateV2::GetTNInt3);
                self.cdrom_queue_response(NOP_TIMING);
            }
            // GetTD - Command 14h,track --> INT3(stat,mm,ss) ;BCD
            0x14 => {
                let n = self.cdrom.get_param::<Bcd>();
                self.cdrom_continue_cmd(CommandStateV2::GetTDInt3(n));
                self.cdrom_queue_response(NOP_TIMING);
            }
            // `SeekL` - Command 15h --> INT3(stat) --> INT2(stat)
            0x15 => {
                let seek_time = self.cdrom.drive.current_seek_time();
                tracing::info!(target: "cdrom", "SeekL\t\t{} seek time: {}ms", self.cdrom.drive.cursor.to_mss::<u8>(), cycles_to_ms(seek_time));
                self.cdrom_continue_cmd(CommandStateV2::SeekLInt3);
                self.cdrom_queue_response(NOP_TIMING);
            }
            // test
            0x19 => {
                let Some(sub) = self.cdrom.drain_params().next() else {
                    self.cdrom_end_cmd();
                    return;
                };
                match sub {
                    // 20h INT3(yy,mm,dd,ver) Get cdrom BIOS date/version (yy,mm,dd,ver)
                    0x20 => {
                        self.cdrom_continue_cmd(CommandStateV2::TestHC05BiosDateInt3);
                        self.cdrom_queue_response(NOP_TIMING);
                    }
                    _ => {
                        tracing::warn!(
                            "todo(cdrom): cmd 0x19 (test) unhandled sub value: {}",
                            hex(sub)
                        );
                        self.cdrom_end_cmd();
                    }
                }
            }
            // GetID
            0x1a => {
                tracing::info!("0x1a GetID");
                self.cdrom_continue_cmd(CommandStateV2::GetIdInt3_5);
                self.cdrom_queue_response(NOP_TIMING);
            }

            // ReadTOC - Command 1Eh --> INT3(stat) --> INT2(stat)
            0x1e => {
                tracing::info!("ReadTOC");
                self.cdrom_continue_cmd(CommandStateV2::ReadTOCInt3);
                self.cdrom_queue_response(INIT_TIMING);
            }

            cmd => todo!("cdrom cmd not implemented: {}", hex(cmd)),
            // _ => {}
        }
    }
    fn cdrom_end_cmd(&mut self) {
        self.cdrom.drive.cmd_state_v2 = CommandStateV2::Idle;
        self.cdrom.status.set_busy_status(false);
    }
    fn cdrom_continue_cmd(&mut self, state: CommandStateV2) {
        self.cdrom.drive.cmd_state_v2 = state;
    }
    fn cdrom_queue_response(&mut self, in_cycles: u32) {
        self.cdrom.ev_id = self.cdrom.ev_id.wrapping_add(1);
        self.evque.schedule(
            EventKind::CdromResponse(self.cdrom.ev_id),
            EventId::default(),
            u64::from(in_cycles),
        );
    }

    pub(crate) fn handle_ev_cdrom_response(&mut self, _: EvCtx, id: u32) {
        if id < self.cdrom.ev_id {
            return;
        }
        self.cdrom_proc_cmd_response();
    }

    #[allow(clippy::match_same_arms)]
    fn cdrom_proc_cmd_response(&mut self) {
        let status = self.cdrom.drive.status_code;
        match self.cdrom.drive.cmd_state_v2 {
            CommandStateV2::Idle => {}

            CommandStateV2::Nop => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::GetIdInt3_5 => match self.cdrom.drive.drive_status {
                DriveStatus::LidOpen => {
                    self.cdrom_send_response_v2(int5([0x11, 0x80]));
                    self.cdrom_end_cmd();
                }
                DriveStatus::SpinUp => {
                    self.cdrom_send_response_v2(int5([0x01, 0x80]));
                    self.cdrom_end_cmd();
                }
                DriveStatus::DetectBusy => {
                    self.cdrom_send_response_v2(int5([0x03, 0x80]));
                    self.cdrom_end_cmd();
                }
                _ => {
                    self.cdrom_send_response_v2(int3(status));
                    self.cdrom_queue_response(GETID_RES_2_TIMING);
                    self.cdrom_continue_cmd(CommandStateV2::GetIdInt2_5);
                }
            },
            CommandStateV2::GetIdInt2_5 => {
                match self.cdrom.drive.drive_status {
                    DriveStatus::NoDisk => {
                        self.cdrom_send_response_v2(int5([0x80, 0x40]));
                        self.cdrom_end_cmd();
                    }
                    DriveStatus::AudioDisk => todo!(),
                    // INT3(stat), INT2(02h,00h, 20h,00h, 53h,43h,45h,4xh)
                    DriveStatus::LicensedMode2 => {
                        self.cdrom_send_response_v2(int2([
                            0x02, 0x00, 0x20, 0x00, 0x53, 0x43, 0x45, 0x49,
                        ]));
                        self.cdrom_end_cmd();
                    }
                    _ => unreachable!(),
                }
            }

            CommandStateV2::TestHC05BiosDateInt3 => {
                self.cdrom_send_response_v2(int3(*self.cdrom.ver));
                self.cdrom_end_cmd();
            }

            CommandStateV2::ReadTOCInt3 => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_continue_cmd(CommandStateV2::ReadTOCInt2);
                self.cdrom_queue_response(INIT_TIMING);
            }
            CommandStateV2::ReadTOCInt2 => {
                self.cdrom_send_response_v2(int2(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::SetlocInt3(mss) => {
                self.cdrom.drive.setloc(mss);
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::SeekLInt3 => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_continue_cmd(CommandStateV2::SeekLInt2);
                let seek_time = self.cdrom.drive.current_seek_time();
                self.cdrom_queue_response(seek_time as u32);
                self.cdrom.drive.status_code.set_spindle_mot(true);
                self.cdrom.drive.status_code.reset_state();
                self.cdrom.drive.status_code.set_seek(true);
                self.cdrom.drive.seek_to_cursor();
            }
            CommandStateV2::SeekLInt2 => {
                self.cdrom_send_response_v2(int2(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::SetModeInt3(setmode) => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_end_cmd();
                debug_assert!(!setmode.xa_adpcm(), "xa-adpcm not yet implemented");
                self.cdrom.drive.setmode(setmode);
            }

            CommandStateV2::PauseInt3 => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_continue_cmd(CommandStateV2::PauseInt2);
                match self.cdrom.drive.mode.speed() {
                    SetModeSpeed::Normal => {
                        self.cdrom_queue_response(PAUSE_1X_TIMING);
                    }
                    SetModeSpeed::Double => {
                        self.cdrom_queue_response(PAUSE_2X_TIMING);
                    }
                }
            }
            CommandStateV2::PauseInt2 => {
                self.cdrom.drive.pause();
                self.cdrom_send_response_v2(int2(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::DemuteInt3 => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::ReadNInt3 => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_continue_cmd(CommandStateV2::ReadNInt1);
                self.cdrom.status.set_busy_status(false);

                let mut cycles = self.cdrom.drive.sector_cycles();
                let seek_time = self.cdrom.drive.current_seek_time();
                if self.cdrom.drive.setloc_armed {
                    cycles += seek_time;
                }
                self.cdrom.drive.seek_to_cursor();

                self.cdrom_queue_response(cycles as u32);
            }
            CommandStateV2::ReadNInt1 => {
                let cycles_per_sector = self.cdrom.drive.sector_cycles();
                self.cdrom_send_response_v2(int1(status));
                self.cdrom_continue_cmd(CommandStateV2::ReadNInt1);
                self.cdrom_queue_response(cycles_per_sector as u32);

                self.cdrom
                    .drive
                    .request_data(&mut self.cdrom.status, &mut self.cdrom.data_fifo);
            }

            CommandStateV2::InitInt3 => {
                self.cdrom_send_response_v2(int3(status));
                self.cdrom_continue_cmd(CommandStateV2::InitInt2);
                let spinup_time = self.cdrom.drive.spinup_time();
                self.cdrom_queue_response(spinup_time as u32 + INIT_TIMING);
            }
            CommandStateV2::InitInt2 => {
                self.cdrom.drive.mode = SetMode::new_with_raw_value(0x20);
                self.cdrom.drive.status_code.reset_state();
                self.cdrom.drive.spin_up();
                self.cdrom.drive.cursor = CdromCursor::default();
                self.cdrom_send_response_v2(int2(status));
                self.cdrom_end_cmd();
            }

            CommandStateV2::GetTNInt3 => {
                let cue_sheet = self
                    .cdrom
                    .drive
                    .get_toc()
                    .expect("todo: called GetTN without a disc. error int5");
                let first = 1;
                let last = cue_sheet.tracks.last().map_or(1, |track| track.number);

                tracing::info!(target: "cdrom", "GetTN\t\t{first},{last}");

                let first = Bcd::from_binary(first).raw_value();
                let last = Bcd::from_binary(last).raw_value();

                self.cdrom_send_response_v2(int3([status.raw_value(), first, last]));
                self.cdrom_end_cmd();
            }

            CommandStateV2::GetTDInt3(n) => {
                let cue_sheet = self
                    .cdrom
                    .drive
                    .get_toc()
                    .expect("todo: called GetTN without a disc. error int5");
                let idx = n.unpack();

                let track = if idx == 0 {
                    cue_sheet.tracks.last().expect("todo: no tracks on disc")
                } else {
                    &cue_sheet.tracks[usize::from(idx) - 1]
                };

                let min = track.index_list[0].min;
                let sec = track.index_list[0].second;

                tracing::info!(target: "cdrom", "GetTD\t\t{min:02},{sec:02}");

                let min = Bcd::from_binary(track.index_list[0].min);
                let sec = Bcd::from_binary(track.index_list[0].second);

                self.cdrom_send_response_v2(int3([
                    status.raw_value(),
                    min.raw_value(),
                    sec.raw_value(),
                ]));
                self.cdrom_end_cmd();
            }
        }
    }
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
#[derive(Endecode)]
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

impl StatusCode {
    pub fn reset_state(&mut self) {
        self.set_read(false);
        self.set_seek(false);
        self.set_play(false);
    }
}

impl Default for StatusCode {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<A: Allocator> CDRomState<A> {
    fn drain_params(&mut self) -> impl Iterator<Item = u8> {
        iter::from_fn(|| self.param_fifo.pop_front())
    }

    fn get_param<T: From<u8>>(&mut self) -> T {
        self.param_fifo.pop_front().unwrap_or_default().into()
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
#[derive(Default, Endecode)]
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

const impl From<u8> for SetMode {
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

impl SetModeSectSize {
    pub const fn len(self) -> usize {
        match self {
            SetModeSectSize::DataOnly0x800 => 0x800,
            SetModeSectSize::Whole0x924 => 0x924,
        }
    }
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum SetModeSpeed {
    Normal = 0x0,
    Double = 0x1,
}
