use alloc::collections::VecDeque;
use core::alloc::Allocator;
use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

use smallvec::{SmallVec, smallvec};

use crate::Emu;
use crate::cpu::Cpu;
use crate::io::cdrom::cdrom_cmds::{Response, SetMode, SetModeSpeed, StatusCode};
use crate::io::cdrom::cdrom_format::{
    CdromCursor, CueFormat, CueFormatParseErr, Mss, SECTOR_USER_SIZE,
};
use crate::io::cdrom::{CDRomStatusReg, CdromScheduler, DriveStatus};
use crate::io::evque::EventId;

use super::cdrom_cmds::SetModeSectSize;

#[derive(Default, derive_more::Debug)]
pub struct CdromDrive {
    pub cursor:        CdromCursor,
    setloc_armed:      bool,
    pub status_code:   StatusCode,
    pub drive_status:  DriveStatus,
    pub mode:          SetMode,
    pub drive_state:   DriveState,
    pub command_state: CommandState,
    disc:              Option<Disc>,
    host_disc_err:     Option<std::io::Error>,

    open_disc_state: Option<OpenDiscFSM>,
}

impl Clone for CdromDrive {
    fn clone(&self) -> Self {
        Self {
            cursor:          self.cursor,
            status_code:     self.status_code,
            drive_status:    self.drive_status.clone(),
            mode:            self.mode,
            drive_state:     self.drive_state.clone(),
            command_state:   self.command_state.clone(),
            disc:            None,
            host_disc_err:   None,
            open_disc_state: None,
            setloc_armed:    false,
        }
    }
}

#[derive(Default, derive_more::Debug, Clone)]
pub enum DriveState {
    #[default]
    Idle,
    ReadN(EventId),
    SeekL(EventId),
}

#[derive(Default, derive_more::Debug, Clone)]
pub(super) enum CommandState {
    #[default]
    Idle,
    Responding(SmallVec<[EventId; 2]>),
}

const CYCLES_PER_BYTE: u64 = Cpu::CLOCK as u64 / (SECTOR_USER_SIZE as u64 * 75);
const CYCLES_PER_BYTE_2X: u64 = CYCLES_PER_BYTE / 2;
const CYCLES_PER_WORD: u64 = CYCLES_PER_BYTE * 4;

impl CdromDrive {
    pub fn setloc<T>(&mut self, mss: Mss<T>)
    where
        u8: const From<T>,
    {
        self.cursor = CdromCursor::from_mss(mss);
        self.setloc_armed = true;
        tracing::info!("setloc: lba={}", self.cursor.lba)
    }

    pub fn seek_to_cursor(&mut self) {
        self.setloc_armed = false;
        if let Some(disc) = &mut self.disc {
            let res = disc.seek(self.cursor);
            self.host_disc_err = res.err();
        }
    }

    pub fn setmode(&mut self, setmode: SetMode) {
        let old_mode = self.mode;
        self.mode = setmode;

        if setmode.ignore_bit() {
            self.mode.set_sect_size(old_mode.sect_size());
        }
    }

    pub fn pause(&mut self) {
        tracing::info!("pause drive");
        self.drive_state = DriveState::Idle;
        self.status_code.reset_state();
    }

    pub fn run<A: Allocator + Copy>(&mut self, scheduler: &mut CdromScheduler<'_, A>) {
        match self.drive_state {
            DriveState::Idle => {}
            DriveState::ReadN(id) => {
                self.status_code.reset_state();
                self.status_code.set_read(true);
                let cycles_per_sector = self.sector_cycles();

                let int1 = Response::new(
                    super::HInt::Int1DataReady,
                    smallvec![self.status_code.raw_value()],
                    false,
                );
                let int1 = scheduler.responses.insert(int1);
                self.drive_state = DriveState::ReadN(int1);
                self.cmd_subscribe_to(int1);

                scheduler.evque.schedule(
                    |emu, ctx| {
                        let DriveState::ReadN(current_id) = emu.cdrom.drive.drive_state else {
                            return;
                        };
                        if current_id != ctx.id {
                            return;
                        }
                        let res = emu.cdrom.responses.remove(current_id);
                        emu.cdrom_send_response(res.expect("event lost"));
                        emu.cdrom
                            .drive
                            .request_data(&mut emu.cdrom.status, &mut emu.cdrom.data_fifo);
                        emu.cdrom.drive.run(&mut CdromScheduler {
                            id:        ctx.id,
                            evque:     &mut emu.evque,
                            responses: &mut emu.cdrom.responses,
                        });
                    },
                    int1,
                    cycles_per_sector,
                );
            }
            DriveState::SeekL(id) => {
                if id == scheduler.id {
                    self.drive_state = DriveState::Idle;
                    self.status_code.reset_state();
                }
            }
        }
    }

    pub fn request_data(&mut self, status: &mut CDRomStatusReg, result_fifo: &mut VecDeque<u8>) {
        status.set_data_req(true);
        if let Some(disc) = &mut self.disc {
            // TODO: better reporting
            tracing::info!("ReadN\t{}", self.cursor.to_mss::<u8>());

            let mut bytes = match disc.readn::<SECTOR_USER_SIZE>(&mut self.cursor) {
                Ok(res) => res,
                Err(err) => {
                    self.host_disc_err = Some(err);
                    return;
                }
            };

            let sector = match self.mode.sect_size() {
                SetModeSectSize::DataOnly0x800 => &bytes[0x18..0x18 + 0x800],
                SetModeSectSize::Whole0x924 => &bytes[0xc..],
            };
            result_fifo.extend(sector);
        }
    }

    pub(super) fn set_command_state(&mut self, state: CommandState) {
        self.command_state = state;
    }

    pub(super) fn cmd_clear(&mut self) {
        self.set_command_state(CommandState::Idle);
    }

    pub(super) fn cmd_subscribe_to(&mut self, id: EventId) {
        match &mut self.command_state {
            CommandState::Idle => self.command_state = CommandState::Responding(smallvec![id]),
            CommandState::Responding(res) => res.push(id),
        }
    }

    fn sector_cycles(&self) -> u64 {
        let mult = match self.mode.speed() {
            SetModeSpeed::Normal => CYCLES_PER_BYTE,
            SetModeSpeed::Double => CYCLES_PER_BYTE_2X,
        };
        self.mode.sect_size().len() as u64 * mult
    }
}

impl CommandState {
    pub(super) fn responding(res: impl IntoIterator<Item = EventId>) -> Self {
        let res = SmallVec::from_iter(res);
        Self::Responding(res)
    }
}

#[derive(derive_more::Debug)]
pub enum DiscReader {
    Streamed(StreamedDiskReader),
    InMemory(InMemoryDiskReader),
}

#[derive(Default, derive_more::Debug, Clone)]
pub struct InMemoryDiskReader {
    #[debug(skip)]
    buf:    Box<[u8]>,
    cursor: u64,
}

impl InMemoryDiskReader {
    fn seek(&mut self, to: u64) -> Result<(), std::io::Error> {
        self.cursor = to;
        Ok(())
    }
}

trait DiscFile: Read + Seek + Send + Sync {}
impl<T> DiscFile for T where T: Read + Seek + Send + Sync {}

#[derive(derive_more::Debug)]
pub struct StreamedDiskReader {
    #[debug(skip)]
    reader: BufReader<Box<dyn DiscFile>>,
}

impl DiscReader {
    pub fn seek(&mut self, to: u64) -> std::io::Result<()> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.seek(to),
            DiscReader::InMemory(in_memory_disk_reader) => in_memory_disk_reader.seek(to),
        }
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        cursor: &mut CdromCursor,
    ) -> std::io::Result<[u8; BYTES]> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.readn(cursor),
            DiscReader::InMemory(_in_memory_disk_reader) => todo!(),
        }
    }
}

impl StreamedDiskReader {
    fn seek(&mut self, to: u64) -> std::io::Result<()> {
        self.reader.seek(std::io::SeekFrom::Start(to))?;
        Ok(())
    }

    fn readn<const BYTES: usize>(
        &mut self,
        cursor: &mut CdromCursor,
    ) -> std::io::Result<[u8; BYTES]> {
        let mut buf = [0u8; BYTES];
        tracing::info!(readn_stream_pos = self.reader.stream_position().unwrap());
        self.reader.read_exact(&mut buf)?;
        let old_cursor = *cursor;

        cursor.lba += (BYTES / SECTOR_USER_SIZE) as u32;
        cursor.byte += (BYTES % SECTOR_USER_SIZE) as u32;

        tracing::info!("lba={}->{}", old_cursor.lba, cursor.lba);

        Ok(buf)
    }
}

#[derive(derive_more::Debug)]
pub enum Disc {
    CueBin(CueFormat, DiscReader),
    Raw(DiscReader),
}

impl Disc {
    pub fn seek(&mut self, to: CdromCursor) -> std::io::Result<()> {
        let (padding, reader) = match self {
            Disc::CueBin(cue_format, disc_reader) => (
                u32::from(cue_format.index_list[0].second) * 75 * SECTOR_USER_SIZE as u32,
                disc_reader,
            ),
            Disc::Raw(disc_reader) => (0, disc_reader),
        };
        let byte = padding + to.to_byte();
        reader.seek(u64::from(byte))
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        cursor: &mut CdromCursor,
    ) -> std::io::Result<[u8; BYTES]> {
        match self {
            Disc::CueBin(_cue_format, disc_reader) => disc_reader.readn(cursor),
            Disc::Raw(disc_reader) => disc_reader.readn(cursor),
        }
    }
}

#[derive(Debug)]
pub enum OpenDiscFSM {
    NeedBin(CueFormat),
    Done,
}

#[derive(thiserror::Error, Debug)]
pub enum OpenDiscErr {
    #[error("fs: {0}")]
    IOErr(#[from] std::io::Error),
    #[error(transparent)]
    CueParseErr(#[from] CueFormatParseErr),
    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),
}

impl<A: Allocator + Copy> Emu<A> {
    pub fn open_disc(
        &mut self,
        path: impl AsRef<Path>,
        streamed: bool,
    ) -> Result<OpenDiscFSM, OpenDiscErr> {
        let path = path.as_ref();
        match path.extension().map(|e| e.to_string_lossy()).as_deref() {
            Some("cue" | "CUE") => {
                let mut format = fs::File::open(path)?;
                let mut buf = String::new();
                let n = format.read_to_string(&mut buf)?;
                let buf = &buf[..n];
                let cue = buf.parse::<CueFormat>()?;
                Ok(OpenDiscFSM::NeedBin(cue))
            }
            _ => self
                .cdrom
                .drive
                .open_disc_bin(path, streamed)
                .map(|disc_reader| {
                    self.cdrom.drive.disc = Some(Disc::Raw(disc_reader));
                    OpenDiscFSM::Done
                }),
        }
    }

    pub fn advance_open_disc(
        &mut self,
        original_path: impl AsRef<Path>,
        fsm: OpenDiscFSM,
        streamed: bool,
    ) -> Result<(), OpenDiscErr> {
        match fsm {
            OpenDiscFSM::NeedBin(cue_format) => {
                let bin_name = &cue_format.filename;
                let original_path = original_path.as_ref();
                let path = original_path
                    .parent()
                    .ok_or_else(|| OpenDiscErr::InvalidPath(original_path.to_owned()))?
                    .to_owned();
                let path = path.join(bin_name);
                let reader = self.cdrom.drive.open_disc_bin(path, streamed)?;
                self.cdrom.drive.disc = Some(Disc::CueBin(cue_format, reader));
                Ok(())
            }
            OpenDiscFSM::Done => Ok(()),
        }
    }
}

impl CdromDrive {
    fn open_disc_bin(
        &mut self,
        path: impl AsRef<Path>,
        streamed: bool,
    ) -> Result<DiscReader, OpenDiscErr> {
        // TODO: detect audio CD
        self.drive_status = DriveStatus::LicensedMode2;
        match streamed {
            true => {
                let file = fs::File::open(path)?;
                Ok(DiscReader::Streamed(StreamedDiskReader {
                    reader: BufReader::new(Box::new(file)),
                }))
            }
            false => {
                let mut file = fs::File::open(path)?;
                let mut buf = Vec::new();
                let n = file.read_to_end(&mut buf)?;
                buf.truncate(n);
                Ok(DiscReader::InMemory(InMemoryDiskReader {
                    buf:    buf.into_boxed_slice(),
                    cursor: 0,
                }))
            }
        }
    }
}
