use alloc::collections::VecDeque;
use core::alloc::Allocator;
use core::hint::cold_path;
use core::marker::PhantomData;
use core::{cmp, fmt, io};
use pchan_utils::{Hex, hex_pref};

// FIXME: replace with non-std compatible versions
use std::io::{BufReader, Read, Seek};

use tracing::Level;

use smallvec::{SmallVec, smallvec};

use crate::Emu;
use crate::cpu::Cpu;
use crate::io::cdrom::cdrom_cmds::{CommandStateV2, SetMode, SetModeSpeed, StatusCode};
use crate::io::cdrom::cdrom_format::{
    self, Bcd, CdromCursor, CueFormatParseErr, CueParser, CueParserState, CueSheet, Mss,
    SECTOR_USER_SIZE, mss_to_lba,
};
use crate::io::cdrom::{CDRomStatusReg, DriveStatus};
use crate::io::evque::{EventId, EventKind, Evque};

use super::cdrom_cmds::SetModeSectSize;

#[derive(derive_more::Debug, Default)]
pub struct CdromDrive {
    pub cursor:        CdromCursor,
    pub current_track: Option<u8>,
    pub toc:           Option<CueSheet>,
    pub old_cursor:    CdromCursor,
    pub setloc_armed:  bool,
    pub current_speed: SetModeSpeed,
    pub status_code:   StatusCode,
    pub drive_status:  DriveStatus,
    pub mode:          SetMode,
    pub drive_state:   DriveState,
    pub command_state: CommandState,
    pub cmd_state_v2:  CommandStateV2,
    pub(super) disc:   Option<Disc>,
    host_disc_err:     Option<DiscError>,
}

#[derive(thiserror::Error, Debug)]
enum DiscError {
    #[error(transparent)]
    OpenDisc(#[from] OpenDiscErr),
    #[error(transparent)]
    AudioPlayback(#[from] AudioPlaybackErr),
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl Clone for CdromDrive {
    fn clone(&self) -> Self {
        Self {
            cursor:        self.cursor,
            current_track: None,
            old_cursor:    self.old_cursor,
            status_code:   self.status_code,
            drive_status:  self.drive_status.clone(),
            mode:          self.mode,
            drive_state:   self.drive_state.clone(),
            command_state: self.command_state.clone(),
            disc:          None,
            host_disc_err: None,
            setloc_armed:  false,
            cmd_state_v2:  CommandStateV2::Idle,
            toc:           None,
            current_speed: SetModeSpeed::Normal,
        }
    }
}

#[derive(Default, derive_more::Debug, Clone)]
pub enum DriveState {
    #[default]
    Idle,
    ReadN(EventId),
    SeekL(EventId),
    Playing(EventId),
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

pub type Lba = u64;

#[derive(Debug, Clone, Copy)]
pub enum SectorKind {
    DataSector,
    CDDASector,
}

impl SectorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            SectorKind::DataSector => "DataSector",
            SectorKind::CDDASector => "CDDASector",
        }
    }
}

impl fmt::Display for SectorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

pub trait CDRomSink {
    fn clear(&mut self);
    fn write(&mut self, data: &[u8]);
}

impl CDRomSink for VecDeque<u8> {
    fn clear(&mut self) {
        self.clear();
    }

    fn write(&mut self, data: &[u8]) {
        self.extend(data);
    }
}

impl CDRomSink for &mut [u8] {
    fn clear(&mut self) {
        self.fill(0);
    }

    fn write(&mut self, data: &[u8]) {
        self.copy_from_slice(data);
    }
}

impl<const N: usize> CDRomSink for [u8; N] {
    fn clear(&mut self) {
        self.fill(0);
    }

    fn write(&mut self, data: &[u8]) {
        self.copy_from_slice(data);
    }
}

impl CdromDrive {
    pub fn setloc<T>(&mut self, mss: Mss<T>)
    where
        u8: const From<T>,
    {
        self.old_cursor = self.cursor;
        self.cursor = CdromCursor::from_mss(mss);
        self.setloc_armed = true;
    }

    pub fn seek_to_cursor(&mut self) {
        self.spin_up();
        self.setloc_armed = false;
        if let Some(disc) = &mut self.disc {
            let res = disc.seek(self.cursor);
            self.host_disc_err = res.err().map(|err| err.into());
        }
    }

    pub fn setmode(&mut self, setmode: SetMode) {
        let old_mode = self.mode;
        self.mode = setmode;

        if setmode.ignore_bit() {
            self.mode.set_sect_size(old_mode.sect_size());
        }
    }

    pub(crate) fn change_speed_time(&self, to_speed: SetModeSpeed) -> u64 {
        if self.current_speed == to_speed {
            0
        } else {
            u64::from(Cpu::CLOCK) * 7 / 10
        }
    }

    pub(crate) fn change_speed_to_mode(&mut self) -> u64 {
        let time = self.change_speed_time(self.mode.speed());
        self.current_speed = self.mode.speed();
        time
    }

    pub fn pause(&mut self) {
        self.drive_state = DriveState::Idle;
        self.status_code.reset_state();
    }

    pub fn request_sector(
        &mut self,
        sector_kind: SectorKind,
        status: &mut CDRomStatusReg,
        data_fifo: &mut dyn CDRomSink,
    ) {
        // fifo can only hold up to one sector
        data_fifo.clear();

        let mut bytes = [0u8; SECTOR_USER_SIZE];
        self.request_raw_sector(sector_kind, status, &mut bytes);

        let sector = match self.mode.sect_size() {
            SetModeSectSize::DataOnly0x800 => &bytes[0x18..0x18 + 0x800],
            SetModeSectSize::Whole0x924 => &bytes[0xc..],
        };

        if tracing::enabled!(Level::TRACE) {
            let mut first32bytes: [Hex<false>; 32] = [hex_pref(0); _];
            for (src, dest) in sector.iter().copied().zip(first32bytes.iter_mut()) {
                *dest = hex_pref(src);
            }
            tracing::debug!(
                target: "cdrom",
                "DataSector\t{} first 32 bytes: {first32bytes:?}",
                self.cursor.to_mss::<u8>(),
            );
        }

        data_fifo.write(sector);
    }

    pub fn request_raw_sector(
        &mut self,
        sector_kind: SectorKind,
        status: &mut CDRomStatusReg,
        sink: &mut dyn CDRomSink,
    ) {
        status.set_data_req(true);

        if let Some(disc) = &mut self.disc {
            tracing::info!(
                target: "cdrom",
                "{sector_kind:<14.14}{cursor} lba={lba} size={size}",
                cursor = self.cursor.to_mss::<u8>(),
                lba = self.cursor.lba,
                size = self.mode.sect_size().raw_value(),
            );

            let bytes = match disc.readn::<SECTOR_USER_SIZE>(&mut self.cursor) {
                Ok(res) => res,
                Err(err) => {
                    self.host_disc_err = Some(err.into());
                    return;
                }
            };

            sink.write(&bytes);
        }
    }

    fn audio_playback_time(&self) -> u64 {
        match self.mode.speed() {
            // 44100 Hz / 75 Hz
            SetModeSpeed::Normal => spu::CLOCK_CYCLES,
            // 44100 Hz / (2 * 75 Hz)
            SetModeSpeed::Double => spu::CLOCK_CYCLES / 2,
        }
    }

    pub(crate) fn play_audio(&mut self, evque: &mut Evque) -> Result<(), AudioPlaybackErr> {
        if self.cursor.to_mss::<u8>().min > 74 {
            return Err(AudioPlaybackErr::ReachedEnd);
        }
        // TODO: might need to sync with the spu
        evque.schedule(EventKind::CdromAudioPlayback, 0, self.audio_playback_time());
        self.drive_state = DriveState::Playing(0);
        Ok(())
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

    pub(super) fn spin_up(&mut self) {
        self.status_code.set_spindle_mot(true);
    }

    pub fn sector_cycles(&self) -> u64 {
        let mult = match self.mode.speed() {
            SetModeSpeed::Normal => CYCLES_PER_BYTE,
            SetModeSpeed::Double => CYCLES_PER_BYTE_2X,
        };
        self.mode.sect_size().len() as u64 * mult
    }

    pub fn spinup_time(&self) -> u64 {
        if self.status_code.spindle_mot() {
            0
        } else {
            u64::from(Cpu::CLOCK)
        }
    }

    pub fn seek_time(&self, from_lba: u32, to_lba: u32) -> u64 {
        const T: f64 = 22_000.0;
        fn lba_to_track(x: u32) -> f64 {
            (-9.0 * T + f64::sqrt(81.0 * T.powi(2) + 26.0 * T * f64::from(x))) / 13.0
        }

        fn sectors_per_track(t: u32) -> f64 {
            13.0 * (f64::from(t) / 22_000.0) + 9.0
        }

        let current_track = lba_to_track(from_lba);
        let track_delta = f64::abs(lba_to_track(to_lba) - current_track);

        let lba_diff = u32::abs_diff(from_lba, to_lba);

        let mut cycles = 0u64;

        if !self.status_code.spindle_mot() {
            cycles += self.spinup_time();
        }

        // if from_lba < 250 && to_lba < 250 {
        //     return cycles + u64::from(lba_diff) * (u64::from(Cpu::CLOCK) * 50 / 1000) / 150;
        // }

        if track_delta < 1. {
            let reading_speed = match self.mode.speed() {
                SetModeSpeed::Normal => Cpu::CLOCK / 75,
                SetModeSpeed::Double => Cpu::CLOCK / 150,
            };
            if from_lba <= to_lba {
                cycles += cmp::max(u64::from(lba_diff), 2) * u64::from(reading_speed);
            } else {
                let jump_to = from_lba.saturating_sub(sectors_per_track(from_lba) as u32);
                cycles += u64::from(cmp::max(to_lba - jump_to, 1)) * u64::from(reading_speed);
            }
        } else {
            const MAX_TIME: f64 = f64::from(Cpu::CLOCK) * 0.9;
            const FIXED_COST: f64 = f64::from(Cpu::CLOCK) * 0.1;
            const TIME_PER_TRACK: f64 = (MAX_TIME - FIXED_COST) / T;
            // let sect_diff = u64::from(to_lba.abs_diff(from_lba));
            // (sect_diff * u64::from(Cpu::CLOCK / 75)).max(u64::from(Cpu::CLOCK) / 75)
            cycles += (FIXED_COST + track_delta * TIME_PER_TRACK) as u64
        }

        cycles
    }

    pub fn current_seek_time(&self) -> u64 {
        self.seek_time(self.old_cursor.lba, self.cursor.lba)
    }

    pub fn current_seek_lba_diff(&self) -> u32 {
        self.old_cursor.lba.abs_diff(self.cursor.lba)
    }

    pub fn read_toc(&mut self) -> Option<&CueSheet> {
        let toc = self.disc.as_mut().map(|disc| disc.get_toc());
        self.toc = toc.cloned();
        toc
    }

    pub fn get_play_track_lba(&self, track: Option<Bcd>) -> PlaySeekResult {
        fn get_inner(
            drive: &CdromDrive,
            track: Option<Bcd>,
            toc: Option<&CueSheet>,
        ) -> Option<PlaySeekResult> {
            let track = track?.unpack();
            let toc = toc?;
            let n = toc.tracks.len() as u8;

            let track = if (1..=n).contains(&track) {
                track
            } else if (n + 1..n + 0x99).contains(&track) {
                drive.current_track?
            } else {
                return None;
            };

            let index = toc.index(track, 1);
            let lba = mss_to_lba(index.rel_timestamp);
            let seek_time = drive.seek_time(drive.cursor.lba, lba);

            Some(PlaySeekResult { lba, seek_time })
        }

        match get_inner(self, track, self.toc.as_ref()) {
            Some(result) => result,
            None => {
                let seek_time = self.current_seek_time();
                let lba = self.cursor.lba;
                PlaySeekResult { lba, seek_time }
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum AudioPlaybackErr {
    #[error("reached end of disc")]
    ReachedEnd,
}

#[derive(Default, derive_more::Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaySeekResult {
    pub lba:       u32,
    pub seek_time: u64,
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

pub trait DiscFile: Read + Seek + Send + Sync + cdrom_format::File {}
impl<T> DiscFile for T where T: Read + Seek + Send + Sync + cdrom_format::File {}

#[derive(derive_more::Debug)]
pub struct StreamedDiskReader {
    #[debug(skip)]
    readers:      Vec<BufReader<Box<dyn DiscFile>>>,
    current_file: usize,
}

impl DiscReader {
    pub fn seek(&mut self, toc: &CueSheet, to: CdromCursor) -> io::Result<()> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.seek(toc, to),
            DiscReader::InMemory(in_memory_disk_reader) => in_memory_disk_reader.seek(toc, to),
        }
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        toc: &CueSheet,
        cursor: &mut CdromCursor,
    ) -> io::Result<[u8; BYTES]> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.readn(toc, cursor),
            DiscReader::InMemory(in_memory_disk_reader) => in_memory_disk_reader.readn(toc, cursor),
        }
    }
}

impl StreamedDiskReader {
    fn seek(&mut self, toc: &CueSheet, to: CdromCursor) -> io::Result<()> {
        debug_assert_eq!(
            self.readers.len(),
            toc.files.len(),
            "mismatch between toc and disc file readers."
        );

        let (file, byte) = toc.lba_to_byte(to.lba);
        self.current_file = file;

        let reader = &mut self.readers[self.current_file];

        if let Some(seek_by) = byte.checked_signed_diff(reader.stream_position()?) {
            BufReader::seek_relative(reader, seek_by)
        } else {
            cold_path();

            reader.seek(io::SeekFrom::Start(byte))?;
            Ok(())
        }
    }

    #[tracing::instrument(target = "cdrom", skip(self), err)]
    fn readn<const BYTES: usize>(
        &mut self,
        toc: &CueSheet,
        cursor: &mut CdromCursor,
    ) -> io::Result<[u8; BYTES]> {
        let mut buf = [0u8; BYTES];
        loop {
            let io_result = self.readers[self.current_file].read_exact(&mut buf);
            let err = io_result.as_ref().map_err(|err| err.kind());
            match err {
                Ok(_) => break,
                Err(io::ErrorKind::Interrupted) => {} // non fatal
                // if we reach eof, move to the next file. if there is no next file,
                // return an error.
                Err(io::ErrorKind::UnexpectedEof) => {
                    if self.current_file < self.readers.len() - 1 {
                        self.current_file += 1;
                        break;
                    }
                    io_result?
                }
                Err(_) => io_result?,
            }
        }

        cursor.lba += (BYTES / SECTOR_USER_SIZE) as u32;
        cursor.byte += (BYTES % SECTOR_USER_SIZE) as u32;
        cursor.lba += cursor.byte / SECTOR_USER_SIZE as u32;
        cursor.byte %= SECTOR_USER_SIZE as u32;

        Ok(buf)
    }
}

impl InMemoryDiskReader {
    #[expect(clippy::unnecessary_wraps)]
    fn seek(&mut self, _toc: &CueSheet, to: CdromCursor) -> io::Result<()> {
        // all files are merged into a single buffer so this works fine i think
        self.cursor = to.to_byte();
        Ok(())
    }

    #[expect(clippy::unnecessary_wraps)]
    fn readn<const BYTES: usize>(
        &mut self,
        _toc: &CueSheet,
        cursor: &mut CdromCursor,
    ) -> io::Result<[u8; BYTES]> {
        let mut buf = [0u8; BYTES];

        let buf_cursor = usize::try_from(self.cursor).unwrap();
        let src_end = cmp::min(buf_cursor + BYTES, self.buf.len());
        let slice = &self.buf[buf_cursor..src_end];
        let dest_end = cmp::min(slice.len(), buf.len());

        buf[..dest_end].copy_from_slice(slice);

        cursor.lba += (BYTES / SECTOR_USER_SIZE) as u32;
        cursor.byte += (BYTES % SECTOR_USER_SIZE) as u32;

        Ok(buf)
    }
}

#[derive(derive_more::Debug)]
pub struct Disc {
    toc:    CueSheet,
    reader: DiscReader,
}

impl Disc {
    pub fn seek(&mut self, to: CdromCursor) -> io::Result<()> {
        self.reader.seek(&self.toc, to)
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        cursor: &mut CdromCursor,
    ) -> io::Result<[u8; BYTES]> {
        self.reader.readn(&self.toc, cursor)
    }

    pub fn get_toc(&mut self) -> &CueSheet {
        &self.toc
    }
}

pub struct NeedCue;
pub struct NeedFile;
pub struct Opening;
pub struct Done;

pub struct OpenDiscFSM<'a, T> {
    _typestate: PhantomData<T>,
    cue_parser: Option<CueParserState<'a>>,
    readers:    ReaderBuilder,
    toc:        Option<CueSheet>,

    pub requested_filename: &'a str,
}

enum ReaderBuilder {
    InMemory(Vec<u8>),
    Streamed(Vec<BufReader<Box<dyn DiscFile>>>),
}

pub enum OpenDiscState<'a> {
    Opening(OpenDiscFSM<'a, Opening>),
    NeedCue(OpenDiscFSM<'a, NeedCue>),
    /// Requesting a file is tied to cue sheet parsing, which
    /// i think is fine, but might be a bit weird.
    NeedFile(OpenDiscFSM<'a, NeedFile>),
    Done(OpenDiscFSM<'a, Done>),
}

#[derive(thiserror::Error, Debug)]
pub enum OpenDiscErr {
    #[error("fs: {0}")]
    IOErr(#[from] io::Error),
    #[error(transparent)]
    CueParseErr(#[from] CueFormatParseErr),
    #[error("invalid path: {0}")]
    InvalidPath(String),
}

impl<'a, T> OpenDiscFSM<'a, T> {
    fn next<U>(self) -> OpenDiscFSM<'a, U> {
        OpenDiscFSM {
            _typestate:         PhantomData,
            cue_parser:         self.cue_parser,
            readers:            self.readers,
            toc:                self.toc,
            requested_filename: self.requested_filename,
        }
    }

    fn push_reader(&mut self, mut reader: Box<dyn DiscFile>) -> Result<(), io::Error> {
        match &mut self.readers {
            ReaderBuilder::InMemory(buf) => {
                reader.read_to_end(buf)?;
                Ok(())
            }
            ReaderBuilder::Streamed(buf_readers) => {
                buf_readers.push(BufReader::new(reader));
                Ok(())
            }
        }
    }
}

impl<'a> OpenDiscFSM<'a, Opening> {
    pub fn advance(mut self) -> Result<OpenDiscState<'a>, OpenDiscErr> {
        match self.cue_parser {
            Some(cue_parser) => match cue_parser {
                CueParserState::Parsing(cue_parser) => {
                    let cue_parser = cue_parser.advance()?;
                    self.cue_parser = Some(cue_parser);

                    Ok(OpenDiscState::Opening(self))
                }
                CueParserState::RequestFile(cue_parser) => {
                    self.requested_filename = cue_parser.requested_filename();
                    self.cue_parser = Some(CueParserState::RequestFile(cue_parser));

                    tracing::info!(target:"cdrom", "request file {}", self.requested_filename);
                    Ok(OpenDiscState::NeedFile(self.next()))
                }
                CueParserState::Done(cue_sheet) => {
                    self.toc = Some(cue_sheet);
                    self.cue_parser = None;

                    Ok(OpenDiscState::Done(self.next()))
                }
            },
            None => Ok(OpenDiscState::Done(self.next())),
        }
    }
}

impl<'a> OpenDiscFSM<'a, NeedCue> {
    pub fn advance(mut self, cue_content: &'a str) -> Result<OpenDiscState<'a>, CueFormatParseErr> {
        let parser = CueParser::new(cue_content);
        self.cue_parser = Some(parser.advance()?);

        while let Some(parser) = self.cue_parser.take() {
            match parser {
                CueParserState::RequestFile(cue_parser) => {
                    self.requested_filename = cue_parser.requested_filename();
                    self.cue_parser = Some(CueParserState::RequestFile(cue_parser));

                    tracing::info!(target:"cdrom", "request file {}", self.requested_filename);
                    return Ok(OpenDiscState::NeedFile(self.next()));
                }
                CueParserState::Parsing(cue_parser) => {
                    self.cue_parser = Some(CueParserState::Parsing(cue_parser));
                }
                CueParserState::Done(cue_sheet) => self.toc = Some(cue_sheet),
            }
        }
        Ok(OpenDiscState::Opening(self.next()))
    }
}

impl<'a> OpenDiscFSM<'a, NeedFile> {
    pub fn advance(
        mut self,
        file: Box<dyn DiscFile>,
    ) -> Result<OpenDiscFSM<'a, Opening>, io::Error> {
        match self.cue_parser.take() {
            Some(CueParserState::RequestFile(cue_parser)) => {
                self.requested_filename = cue_parser.requested_filename();
                let cue_parser = cue_parser.advance(&*file);
                self.cue_parser = Some(CueParserState::Parsing(cue_parser));
            }
            Some(CueParserState::Done(cue_sheet)) => {
                self.toc = Some(cue_sheet);
            }
            None => {}
            _ => unreachable!(),
        };

        self.push_reader(file)?;
        Ok(self.next())
    }
}

impl OpenDiscFSM<'_, Done> {
    pub fn commit(self, emu: &mut Emu) {
        let toc = self.toc.unwrap();
        emu.cdrom.drive.toc = Some(toc.clone());
        let reader = match self.readers {
            ReaderBuilder::InMemory(buf) => DiscReader::InMemory(InMemoryDiskReader {
                buf:    buf.into_boxed_slice(),
                cursor: 0,
            }),
            ReaderBuilder::Streamed(buf_readers) => DiscReader::Streamed(StreamedDiskReader {
                readers:      buf_readers,
                current_file: 0,
            }),
        };
        emu.cdrom.drive.drive_status = DriveStatus::LicensedMode2;
        emu.cdrom.drive.disc = Some(Disc { toc, reader });
    }
}

impl Emu {
    pub fn open_disc_with_cue<'a>(
        &mut self,
        cue_contents: &'a str,
        streamed: bool,
    ) -> Result<OpenDiscState<'a>, OpenDiscErr> {
        let readers = match streamed {
            true => ReaderBuilder::Streamed(Vec::new()),
            false => ReaderBuilder::InMemory(Vec::new()),
        };
        let fsm = OpenDiscFSM::<NeedCue> {
            _typestate: PhantomData,
            cue_parser: None,
            readers,
            toc: None,
            requested_filename: "",
        };
        let fsm = fsm.advance(cue_contents)?;
        Ok(fsm)
    }

    #[expect(unused, unreachable_code, clippy::diverging_sub_expression)]
    pub fn open_disc_raw(
        &mut self,
        reader: Box<dyn DiscFile>,
        streamed: bool,
    ) -> Result<(), OpenDiscErr> {
        let reader = self.cdrom.drive.create_raw_disc_reader(reader, streamed)?;
        self.cdrom.drive.disc = Some(Disc {
            toc: todo!("get toc from subch Q or use dummy cue sheet."),
            reader,
        });
        self.cdrom.drive.drive_status = DriveStatus::LicensedMode2;
        Ok(())
    }

    pub(crate) fn handle_ev_cdrom_audio_playback(&mut self) {
        let DriveState::Playing(_) = self.cdrom.drive.drive_state else {
            tracing::warn!(target:"cdrom", "cdda playback dropped.");
            return;
        };

        if self.cdrom.sector_buf.len() < 4 {
            self.cdrom.sector_buf.clear();
            return;
        }

        let samples = self.cdrom.sector_buf.drain(..4);
        let mut sample_buf = [0u8; 4];
        for (src, dest) in samples.zip(sample_buf.iter_mut()) {
            *dest = src;
        }
        let sample_l = i16::from_le_bytes([sample_buf[0], sample_buf[1]]);
        let sample_r = i16::from_le_bytes([sample_buf[2], sample_buf[3]]);

        self.cdrom.audio_sample = (sample_l, sample_r);
        if let Err(err) = self.cdrom.drive.play_audio(&mut self.evque) {
            tracing::warn!(target:"cdrom", %err);
            self.cdrom.drive.host_disc_err = Some(err.into());
        }
    }
}

impl CdromDrive {
    fn create_raw_disc_reader(
        &mut self,
        mut reader: Box<dyn DiscFile>,
        streamed: bool,
    ) -> Result<DiscReader, OpenDiscErr> {
        // TODO: detect audio CD
        self.drive_status = DriveStatus::LicensedMode2;
        match streamed {
            true => Ok(DiscReader::Streamed(StreamedDiskReader {
                readers:      vec![BufReader::new(reader)],
                current_file: 0,
            })),
            false => {
                let mut buf = Vec::new();
                reader.read_to_end(&mut buf)?;
                Ok(DiscReader::InMemory(InMemoryDiskReader {
                    buf:    buf.into_boxed_slice(),
                    cursor: 0,
                }))
            }
        }
    }
}
