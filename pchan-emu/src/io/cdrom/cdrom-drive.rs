use std::collections::VecDeque;
use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use heapless::Deque;
use smallvec::{SmallVec, smallvec};

use crate::Emu;
use crate::cpu::Cpu;
use crate::io::cdrom::cdrom_cmds::{Response, SetMode, SetModeSpeed, StatusCode};
use crate::io::cdrom::cdrom_format::{
    CdromCursor, CueFormat, CueFormatParseErr, Mss, SECTOR_USER_SIZE,
};
use crate::io::cdrom::{CDRomStatusReg, CdromScheduler, DriveStatus};

use super::cdrom_cmds::SetModeSectSize;

#[derive(Default, derive_more::Debug)]
pub struct CdromDrive {
    pub cursor:        CdromCursor,
    pub status_code:   StatusCode,
    pub drive_status:  DriveStatus,
    pub mode:          SetMode,
    drive_state:       DriveState,
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
        }
    }
}

#[derive(Default, derive_more::Debug, Clone)]
enum DriveState {
    #[default]
    Idle,
    ReadN,
}

#[derive(Default, derive_more::Debug, Clone)]
pub(super) enum CommandState {
    #[default]
    Idle,
    Responding(SmallVec<[usize; 2]>),
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
        tracing::info!("setloc at {:?}", self.cursor);
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

    pub fn readn(&mut self) {
        self.drive_state = DriveState::ReadN;
    }

    pub fn pause(&mut self) {
        tracing::info!("pause drive");
        self.drive_state = DriveState::Idle;
        self.status_code.reset_state();
    }

    pub fn run(&mut self, scheduler: &mut CdromScheduler<'_>) {
        match self.drive_state {
            DriveState::Idle => {}
            DriveState::ReadN => {
                self.status_code.reset_state();
                self.status_code.set_read(true);
                let cycles_per_sector = self.sector_cycles();
                scheduler.evque.schedule(
                    |emu, _| {
                        tracing::info!("readn callback");
                        emu.cdrom_send_response(Response::new(
                            super::HInt::Int1DataReady,
                            smallvec![emu.cdrom.drive.status_code.raw_value()],
                            false,
                        ));
                        emu.cdrom
                            .drive
                            .request_data(&mut emu.cdrom.status, &mut emu.cdrom.data_fifo);
                        emu.cdrom.drive.run(&mut CdromScheduler {
                            evque:     &mut emu.evque,
                            responses: &mut emu.cdrom.responses,
                        });
                    },
                    0,
                    cycles_per_sector,
                );
            }
        }
    }

    pub fn request_data(&mut self, status: &mut CDRomStatusReg, result_fifo: &mut VecDeque<u8>) {
        status.set_data_req(true);
        if let Some(disc) = &mut self.disc {
            let (n, bytes) = match disc.readn::<SECTOR_USER_SIZE>(self.mode) {
                Ok(res) => res,
                Err(err) => {
                    self.host_disc_err = Some(err);
                    return;
                }
            };
            if n == 0 {
                return;
            }
            let sector = match self.mode.sect_size() {
                SetModeSectSize::DataOnly0x800 => &bytes[0x18..0x18 + 0x800],
                SetModeSectSize::Whole0x924 => &bytes[..],
            };
            result_fifo.extend(sector);
        }
    }

    pub(super) fn set_command_state(&mut self, state: CommandState) {
        self.command_state = state;
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
    pub(super) fn responding(res: impl IntoIterator<Item = usize>) -> Self {
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
    cursor: CdromCursor,
}

trait DiscFile: Read + Seek + Send + Sync {}
impl<T> DiscFile for T where T: Read + Seek + Send + Sync {}

#[derive(derive_more::Debug)]
pub struct StreamedDiskReader {
    #[debug(skip)]
    reader: BufReader<Box<dyn DiscFile>>,
    cursor: CdromCursor,
}

impl DiscReader {
    fn update_cursor(&mut self, cursor: CdromCursor) {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => {
                streamed_disk_reader.cursor = cursor;
            }
            DiscReader::InMemory(in_memory_disk_reader) => {
                in_memory_disk_reader.cursor = cursor;
            }
        }
    }
    pub fn seek(&mut self, to: u64) -> std::io::Result<()> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.seek(to),
            DiscReader::InMemory(in_memory_disk_reader) => todo!(),
        }
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        mode: SetMode,
    ) -> std::io::Result<(usize, [u8; BYTES])> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.readn(mode),
            DiscReader::InMemory(in_memory_disk_reader) => todo!(),
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
        mode: SetMode,
    ) -> std::io::Result<(usize, [u8; BYTES])> {
        tracing::info!("readn\t{}", self.cursor.to_mss::<u8>());
        let mut buf = [0u8; BYTES];
        let n = self.reader.read(&mut buf)?;

        self.cursor.advance_by(BYTES as u32, mode.sect_size());
        self.reader.seek(std::io::SeekFrom::Current(BYTES as i64))?;
        Ok((n, buf))
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
            Disc::CueBin(cue_format, disc_reader) => {
                (cue_format.index_list[0].second as u32 * 75, disc_reader)
            }
            Disc::Raw(disc_reader) => (0, disc_reader),
        };
        let byte = padding + to.to_byte();
        reader.update_cursor(to);
        reader.seek(byte as u64)
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        mode: SetMode,
    ) -> std::io::Result<(usize, [u8; BYTES])> {
        match self {
            Disc::CueBin(cue_format, disc_reader) => disc_reader.readn(mode),
            Disc::Raw(disc_reader) => todo!(),
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

impl Emu {
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
                .map(|_| OpenDiscFSM::Done),
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
                    cursor: CdromCursor::default(),
                }))
            }
            false => {
                let mut file = fs::File::open(path)?;
                let mut buf = Vec::new();
                let n = file.read_to_end(&mut buf)?;
                buf.truncate(n);
                Ok(DiscReader::InMemory(InMemoryDiskReader {
                    buf:    buf.into_boxed_slice(),
                    cursor: CdromCursor::default(),
                }))
            }
        }
    }
}
