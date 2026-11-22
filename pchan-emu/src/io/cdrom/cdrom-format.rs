use core::marker::PhantomData;
use core::ops::Range;
use core::str::FromStr;
use core::{fmt, mem};
use std::marker::Destruct;

use alloc::io;
use arbitrary_int::prelude::*;
use bitbybit::bitfield;

#[bitfield(u8, debug)]
#[derive(PartialEq, Eq, Default)]
pub struct Bcd {
    /// the least significant digit
    #[bits(0..=3, rw)]
    digit_01: u4,
    /// the most significant digit
    #[bits(4..=7, rw)]
    digit_02: u4,
}

impl fmt::Display for Bcd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}", self.unpack())
    }
}

impl Bcd {
    pub const fn unpack(self) -> u8 {
        self.digit_02().value() * 10 + self.digit_01().value()
    }

    pub fn from_binary(num: u8) -> Self {
        let digit_1 = num % 10;
        let digit_2 = (num / 10) % 10;
        Bcd::default()
            .with_digit_01(digit_1.as_())
            .with_digit_02(digit_2.as_())
    }
}

const impl From<Bcd> for u8 {
    fn from(val: Bcd) -> Self {
        val.unpack()
    }
}

const impl From<u8> for Bcd {
    fn from(value: u8) -> Self {
        Self::new_with_raw_value(value)
    }
}

#[derive(Default, derive_more::Debug, Clone, Copy)]
pub struct CdromCursor {
    pub lba:  u32,
    pub byte: u32,
}

/// (minute, second, sector) tuple
#[derive(Debug, Clone, Copy, derive_more::Display, Default, PartialEq, Eq)]
#[display("{min:02}:{sec:02}:{sect:02}")]
pub struct Mss<T> {
    pub min:  T,
    pub sec:  T,
    pub sect: T,
}

impl<T> Mss<T> {
    pub fn new(min: T, sec: T, sect: T) -> Self {
        Self { min, sec, sect }
    }
}

pub const SECTOR_USER_SIZE: usize = 0x930;

pub fn mss_to_lba<T: Into<u8>>(mss: Mss<T>) -> u32 {
    CdromCursor::from_mss(mss).lba
}
pub fn lba_to_mss<T: From<u8>>(lba: u32) -> Mss<T> {
    CdromCursor::new(lba, 0).to_mss()
}

impl CdromCursor {
    pub const fn new(lba: u32, byte: u32) -> Self {
        Self { lba, byte }
    }

    pub fn from_mss<T: Into<u8> + Destruct>(mss: Mss<T>) -> Self {
        let min = u32::from(mss.min.into());
        let sec = u32::from(mss.sec.into());
        let sect = u32::from(mss.sect.into());
        Self {
            lba:  (min * (60 * 75) + sec * 75 + sect),
            byte: 0,
        }
    }

    pub fn to_mss<T: From<u8>>(self) -> Mss<T> {
        let lba = self.lba;

        let sect = (lba % 75) as u8;
        let sec = (lba / 75 % 60) as u8;
        let min = (lba / 75 / 60) as u8;

        Mss {
            min:  T::from(min),
            sec:  T::from(sec),
            sect: T::from(sect),
        }
    }

    pub const fn to_byte(self) -> u64 {
        u64::from(self.lba) * SECTOR_USER_SIZE as u64 + u64::from(self.byte)
    }

    pub const fn advance_by(&mut self, by_sectors: u32, by_bytes: u32) {
        self.lba += by_sectors;
        self.byte += by_bytes;
    }
}

/// TODO: add support for multiple tracks
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueSheet {
    pub files:     Vec<CueFile>,
    pub tracks:    Vec<Track>,
    pub indices:   Vec<CueIndex>,
    /// cd track number -> vector index
    pub tracklist: Vec<TrackIdx>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackKind {
    Audio,
    Mode2_2352,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    pub number:  u8,
    pub kind:    TrackKind,
    pub indices: Range<u8>,
    pub abs_lba: u32,
    pub rel_lba: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CueIndex {
    pub rel_timestamp: Mss<u8>,
    pub abs_timestamp: Mss<u8>,
    pub abs_lba:       u32,
    pub number:        u8,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum CueFormatParseErr {
    #[error("expected `FILE` attribute")]
    ExpectedFile,
    #[error("expected filename in double quotes")]
    ExpectedFilename,
    #[error("expected `TRACK` attribute")]
    ExpectedTrack,
    #[error("expected track number for TRACK")]
    ExpectedTrackNumber,
    #[error("expected track kind (eg. 'AUDIO' or 'MODE2/2352')")]
    ExpectedTrackKind,
    #[error("`INDEX` attribute must be followed by a number")]
    ExpectedIndexNumber,
    #[error("`INDEX` number must be followed by a timestamp")]
    ExpectedTimestamp,
    #[error("io error: {0}")]
    IOError(core::io::ErrorKind),
}

impl From<io::Error> for CueFormatParseErr {
    fn from(value: io::Error) -> Self {
        Self::IOError(value.kind())
    }
}

impl FromStr for TrackKind {
    type Err = CueFormatParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("AUDIO") {
            Ok(Self::Audio)
        } else if s.eq_ignore_ascii_case("MODE2/2352") {
            Ok(Self::Mode2_2352)
        } else {
            Err(CueFormatParseErr::ExpectedTrackKind)
        }
    }
}

pub trait File {
    fn size(&self) -> u64;
}

impl File for std::fs::File {
    fn size(&self) -> u64 {
        self.metadata().map_or_default(|meta| meta.len())
    }
}

pub struct CueParser<'s, T = cue::Parsing> {
    _typestate: PhantomData<T>,
    buf:        &'s str,
    files:      Vec<CueFile>,
    tracks:     Vec<Track>,
    indices:    Vec<CueIndex>,

    parsing:       CueParserItem,
    file_builder:  Option<CueFileBuilder>,
    track_builder: Option<Track>,

    file_lba:  u32,
    track_lba: u32,
    index_lba: u32,

    pub requested_filename: Option<&'s str>,
}

enum CueParserItem {
    None,
    File,
    Track,
    Index,
}

type TrackIdx = u8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueFile {
    filename:  String,
    tracks:    Range<u8>,
    bytes_len: u64,
    lba:       u32,
}

struct CueFileBuilder {
    filename:  String,
    tracks:    Vec<Track>,
    bytes_len: u64,
}

pub(crate) mod cue {
    pub struct Parsing;
    pub struct RequestFile;
}

pub enum CueParserState<'a> {
    /// useful for storing intermediary fsms, but
    /// nothing actually returns it.
    Parsing(CueParser<'a, cue::Parsing>),
    RequestFile(CueParser<'a, cue::RequestFile>),
    Done(CueSheet),
}

impl<'a, T> CueParser<'a, T> {
    pub(crate) fn next<U>(self) -> CueParser<'a, U> {
        CueParser {
            _typestate:         PhantomData,
            buf:                self.buf,
            files:              self.files,
            tracks:             self.tracks,
            file_builder:       self.file_builder,
            track_builder:      self.track_builder,
            parsing:            self.parsing,
            indices:            self.indices,
            requested_filename: self.requested_filename,
            file_lba:           self.file_lba,
            track_lba:          self.track_lba,
            index_lba:          self.index_lba,
        }
    }
}

fn take_until<'a, P: core::str::pattern::Pattern>(str: &mut &'a str, pat: P) -> Option<&'a str> {
    let Some((token, rest)) = str.split_once(pat) else {
        *str = "";
        return None;
    };
    *str = rest;
    Some(token)
}

fn take_until_whitespace<'a>(str: &mut &'a str) -> Option<&'a str> {
    take_until(str, char::is_whitespace)
}

fn take_separated<'a, P: core::str::pattern::Pattern + Copy>(
    str: &mut &'a str,
    sep: P,
) -> Option<&'a str> {
    let _ = take_until(str, sep);

    take_until(str, sep)
}

impl<'a> CueParser<'a, cue::Parsing> {
    pub fn new(buf: &'a str) -> Self {
        CueParser {
            _typestate: PhantomData,
            buf,
            files: Vec::new(),
            tracks: Vec::new(),
            indices: Vec::new(),
            parsing: CueParserItem::None,
            file_builder: None,
            track_builder: None,
            requested_filename: None,
            file_lba: 0,
            track_lba: 0,
            index_lba: 0,
        }
    }

    fn push_file(&mut self) {
        if let Some(fb) = &mut self.file_builder {
            let start = self.tracks.len() as u8;
            let end = start + fb.tracks.len() as u8;
            self.tracks.extend_from_slice(&fb.tracks);

            self.files.push(CueFile {
                filename:  mem::take(&mut fb.filename),
                bytes_len: fb.bytes_len,
                tracks:    start..end,
                lba:       self.file_lba,
            });

            self.file_lba += (fb.bytes_len / SECTOR_USER_SIZE as u64) as u32;
        }
    }

    fn push_track(&mut self) {
        if let Some((tb, fb)) = self.track_builder.take().zip(self.file_builder.as_mut()) {
            self.index_lba = 0;
            fb.tracks.push(tb);
        }
    }

    fn finish(mut self) -> CueSheet {
        self.push_track();
        self.push_file();

        let tracklist = self
            .files
            .iter()
            .flat_map(|f| f.tracks.clone())
            .collect::<Vec<_>>();

        CueSheet {
            files: self.files,
            tracks: self.tracks,
            indices: self.indices,
            tracklist,
        }
    }

    pub fn advance(mut self) -> Result<CueParserState<'a>, CueFormatParseErr> {
        loop {
            self.buf = self.buf.trim_start();
            if self.buf.is_empty() {
                return Ok(CueParserState::Done(self.finish()));
            }

            match self.parsing {
                CueParserItem::None => {
                    let Some(token) = take_until_whitespace(&mut self.buf) else {
                        continue;
                    };

                    if token.eq_ignore_ascii_case("FILE") {
                        self.parsing = CueParserItem::File;
                    } else if token.eq_ignore_ascii_case("TRACK") {
                        self.parsing = CueParserItem::Track;
                    } else if token.eq_ignore_ascii_case("INDEX") {
                        self.parsing = CueParserItem::Index;
                    } else if token.eq_ignore_ascii_case("PREGAP") {
                        todo!("cue parser: handle pregap item");
                    }
                }
                CueParserItem::File => {
                    // push old tracks into file, then push old file
                    self.push_track();
                    self.push_file();

                    self.buf = self.buf.trim_start();
                    let filename = take_separated(&mut self.buf, '"')
                        .ok_or(CueFormatParseErr::ExpectedFilename)?;

                    self.buf = self.buf.trim_start();
                    let _file_type = take_until_whitespace(&mut self.buf);

                    self.file_builder = Some(CueFileBuilder {
                        filename:  filename.to_owned(),
                        tracks:    Vec::new(),
                        bytes_len: 0,
                    });

                    self.parsing = CueParserItem::None;
                    self.requested_filename = Some(filename);
                    return Ok(CueParserState::RequestFile(self.next()));
                }
                CueParserItem::Track => {
                    self.file_builder
                        .as_ref()
                        .ok_or(CueFormatParseErr::ExpectedFile)?;

                    // push old track
                    self.push_track();

                    self.buf = self.buf.trim_start();
                    let number = take_until_whitespace(&mut self.buf)
                        .ok_or(CueFormatParseErr::ExpectedTrackNumber)
                        .and_then(|num| {
                            num.parse::<u8>()
                                .map_err(|_| CueFormatParseErr::ExpectedTrackNumber)
                        })?;
                    self.buf = self.buf.trim_start();
                    let track_kind = take_until_whitespace(&mut self.buf)
                        .and_then(|kind| kind.parse::<TrackKind>().ok())
                        .ok_or(CueFormatParseErr::ExpectedTrackKind)?;

                    let start = self.indices.len() as u8;
                    self.track_builder = Some(Track {
                        number,
                        kind: track_kind,
                        indices: start..start,
                        abs_lba: self.file_lba + self.track_lba,
                        rel_lba: self.track_lba,
                    });

                    self.parsing = CueParserItem::None;
                }
                CueParserItem::Index => {
                    let tb = self
                        .track_builder
                        .as_mut()
                        .ok_or(CueFormatParseErr::ExpectedTrack)?;

                    self.buf = self.buf.trim_start();
                    let number = take_until_whitespace(&mut self.buf)
                        .and_then(|num| num.parse::<u8>().ok())
                        .ok_or(CueFormatParseErr::ExpectedIndexNumber)?;

                    self.buf = self.buf.trim_start();
                    let timestamp = take_until_whitespace(&mut self.buf)
                        .and_then(|t| {
                            let mut t = t.split(':');
                            let min = t.next()?.parse::<u8>().ok()?;
                            let sec = t.next()?.parse::<u8>().ok()?;
                            let sect = t.next()?.parse::<u8>().ok()?;
                            let t = Mss::new(min, sec, sect);
                            Some(t)
                        })
                        .ok_or(CueFormatParseErr::ExpectedTimestamp)?;

                    let abs_lba = self.file_lba + self.track_lba + mss_to_lba(timestamp);
                    let abs_timestamp = lba_to_mss(abs_lba);
                    self.indices.push(CueIndex {
                        rel_timestamp: timestamp,
                        number,
                        abs_timestamp,
                        abs_lba,
                    });
                    self.track_lba += mss_to_lba(timestamp);

                    tb.indices.end += 1;

                    self.parsing = CueParserItem::None;
                }
            }
        }
    }

    fn parse(
        buf: &'a str,
        cb: impl Fn(CueParserStep<'a>) -> Result<CueParser<'a, cue::Parsing>, CueFormatParseErr>,
    ) -> Result<CueSheet, CueFormatParseErr> {
        let mut parser = Self::new(buf);

        loop {
            match parser.advance()? {
                CueParserState::RequestFile(cue_parser) => {
                    parser = cb(CueParserStep::RequestFile(cue_parser))?;
                }
                CueParserState::Done(cue_sheet) => return Ok(cue_sheet),
                CueParserState::Parsing(_) => unreachable!(),
            }
        }
    }
}

enum CueParserStep<'a> {
    RequestFile(CueParser<'a, cue::RequestFile>),
}

impl<'a> CueParser<'a, cue::RequestFile> {
    pub fn requested_filename(&self) -> &'a str {
        self.requested_filename
            .expect("invalid state: requested_filename is None")
    }

    pub fn advance<'f>(mut self, file: &'f dyn File) -> CueParser<'a, cue::Parsing> {
        let fb = self.file_builder.as_mut().expect("no file builder!");
        fb.bytes_len = file.size();
        self.requested_filename = None;
        self.next()
    }
}

pub type FileIdx = usize;

impl CueSheet {
    pub fn track_count(&self) -> u8 {
        self.tracklist.len() as u8
    }
    pub fn index(&self, track: u8, idx: u8) -> &CueIndex {
        debug_assert!(track != 0, "track cannot be 0");

        let track_idx = self.tracklist[usize::from(track - 1)];
        let start = self.tracks[usize::from(track_idx)].indices.start;
        &self.indices[usize::from(idx + start)]
    }

    pub fn lba_to_byte(&self, lba: u32) -> (FileIdx, u64) {
        let file_idx = self
            .files
            .iter()
            .position(|file| file.lba > lba)
            .map_or(self.files.len().saturating_sub(1), |idx| {
                idx.saturating_sub(1)
            });

        let relative_lba = lba - self.files[file_idx].lba;
        let relative_lba = relative_lba.saturating_sub(150);

        let byte = CdromCursor::new(relative_lba, 0).to_byte();

        tracing::info!(?lba, ?relative_lba, ?file_idx);

        (file_idx, byte)
    }
}

/// ```plaintext
///  000h 0Ch  Sync   (00h,FFh,FFh,FFh,FFh,FFh,FFh,FFh,FFh,FFh,FFh,00h)
///  00Ch 4    Header (Minute,Second,Sector,Mode=02h)
///  010h 4    Sub-Header (File, Channel, Submode AND DFh, Codinginfo)
///  014h 4    Copy of Sub-Header
///  018h 800h Data (2048 bytes)
///  818h 4    EDC (checksum across [010h..817h])
///  81Ch 114h ECC (error correction codes)
/// ```
pub struct Form1SectorMetadata {
    sync:      SyncHeader,
    header:    Header,
    subheader: Subheader,
}

struct SyncHeader;

static SYNC_HEADER: &[u8] = &[
    0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
];
const SYNC_HEADER_LEN: usize = SYNC_HEADER.len();

/// `4 Header (Minute,Second,Sector,Mode)`
struct Header {
    mss:  Mss<Bcd>,
    mode: HeaderMode,
}

#[derive(Debug, Clone, Copy)]
enum HeaderMode {
    Mode2 = 0x02,
}

/// `4 Sub-Header (File, Channel, Submode AND DFh, Codinginfo)`
struct Subheader {
    file:    FileNumber,
    channel: ChannelNumber,
    submode: Submode,
    cinfo:   CodingInfo,
}

/// # 3rd Subheader byte - Submode (SM)
///
/// ```plaintext
/// 0   End of Record (EOR) (all Volume Descriptors, and all sectors with EOF)
/// 1   Video     ;\Sector Type (usually ONE of these bits should be set)
/// 2   Audio     ; Note: PSX .STR files are declared as Data (not as Video)
/// 3   Data      ;/
/// 4   Trigger           (for application use)
/// 5   Form2             (0=Form1/800h-byte data, 1=Form2, 914h-byte data)
/// 6   Real Time (RT)
/// 7   End of File (EOF) (or end of Directory/PathTable/VolumeTerminator)
/// ```
#[bitfield(u8)]
pub struct Submode {
    #[bit(0, rw)]
    eor:     bool,
    #[bit(1, rw)]
    video:   bool,
    #[bit(2, rw)]
    audio:   bool,
    #[bit(3, rw)]
    data:    bool,
    #[bit(4, rw)]
    trigger: bool,
    #[bit(5, rw)]
    form2:   bool,
    #[bit(6, rw)]
    rt:      bool,
    #[bit(7, rw)]
    eof:     bool,
}

/// # 4th Subheader byte - Codinginfo (CI)
///
/// When used for Data sectors:
///
/// ```plaintext
///   0-7 Reserved (00h)
/// ```
///
/// When used for XA-ADPCM audio sectors:
///
/// ```plaintext
///   0-1 Mono/Stereo     (0=Mono, 1=Stereo, 2-3=Reserved)
///   2-2 Sample Rate     (0=37800Hz, 1=18900Hz, 2-3=Reserved)
///   4-5 Bits per Sample (0=Normal/4bit, 1=8bit, 2-3=Reserved)
///   6   Emphasis        (0=Normal/Off, 1=Emphasis)
///   7   Reserved        (0)
/// ```
#[bitfield(u8, default = 0x00)]
struct CodingInfo {
    #[bits(0..=1, rw)]
    mono_stereo: u2,
    #[bits(2..=3)]
    sample_rate: XAADPCMSampleRate,
    // TODO: the rest of the fields
}

/// # 1st Subheader byte - File Number (FN)
struct FileNumber(u8);
/// # 2nd Subheader byte - Channel Number (CN)
struct ChannelNumber(u8);

#[cfg(test)]
mod cuetests {

    use crate::io::cdrom::cdrom_format::{
        CueFile, CueFormatParseErr, CueIndex, CueParser, CueParserStep, CueSheet, File, Mss,
        SECTOR_USER_SIZE, Track,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn test_basic_cue() {
        let string = r#"
            FILE "Valkyrie Profile (USA) (Disc 1).bin" BINARY
              TRACK 01 MODE2/2352
                INDEX 01 00:00:00
            "#;
        todo!()
    }

    #[test]
    fn test_2sec_cue() {
        let string = r#"
            FILE "Valkyrie Profile (USA) (Disc 1).bin" BINARY
              TRACK 01 MODE2/2352
                INDEX 01 00:02:00
              TRACK 02 MODE2/2352
                INDEX 01 00:04:00
            "#;
        todo!()
    }

    #[test]
    fn test_new_cue_parser() {
        struct MockFile;

        impl File for MockFile {
            fn size(&self) -> u64 {
                32 * SECTOR_USER_SIZE as u64
            }
        }

        let string = r#"
            FILE "Ridge Racer (Japan) (Track 01).bin" BINARY
            TRACK 01 MODE2/2352
                INDEX 01 00:00:00
            FILE "Ridge Racer (Japan) (Track 02).bin" BINARY
            TRACK 02 AUDIO
                INDEX 00 00:00:00
                INDEX 01 00:02:00
            FILE "Ridge Racer (Japan) (Track 03).bin" BINARY
            TRACK 03 AUDIO
                INDEX 00 00:00:00
                INDEX 01 00:02:00
            "#;

        let result = CueParser::parse(string, |step| match step {
            CueParserStep::RequestFile(cue_parser) => {
                let advance = cue_parser.advance(&MockFile);
                Ok(advance)
            }
        })
        .unwrap();

        println!("{result:#?}");
    }
}
