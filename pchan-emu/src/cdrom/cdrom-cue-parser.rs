use core::marker::PhantomData;
use core::mem;
use core::ops::Range;
use core::str::FromStr;

use alloc::io;

use crate::cdrom::cdrom_format::{CdromCursor, Mss, SECTOR_TOTAL_SIZE, lba_to_mss, mss_to_lba};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueSheet {
    pub files:     Vec<CueFile>,
    pub tracks:    Vec<Track>,
    pub indices:   Vec<CueIndex>,
    /// cd track number -> vector index
    pub tracklist: Vec<TrackIdx>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Audio,
    Mode2_2352,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    pub number:         u8,
    pub kind:           TrackKind,
    pub indices:        Range<u8>,
    pub index_0_pregap: bool,
    pub abs_lba:        u32,
    pub rel_lba:        u32,
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
    #[error("expected file type (eg. BINARY)")]
    ExpectedFileType,
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
    filename:   String,
    tracks:     Range<u8>,
    bytes_len:  u64,
    lba:        u32,
    has_pregap: bool,
}

struct CueFileBuilder {
    filename:   String,
    tracks:     Vec<Track>,
    bytes_len:  u64,
    has_pregap: bool,
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
                filename:   mem::take(&mut fb.filename),
                bytes_len:  fb.bytes_len,
                tracks:     start..end,
                lba:        self.file_lba,
                has_pregap: fb.has_pregap,
            });

            self.file_lba += (fb.bytes_len / SECTOR_TOTAL_SIZE as u64) as u32;
            self.track_lba = 0;
        }
    }

    fn push_track(&mut self) {
        if let Some((tb, fb)) = self.track_builder.take().zip(self.file_builder.as_mut()) {
            if tb.index_0_pregap {
                fb.has_pregap = true;
            }
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
                    let _file_type = take_until_whitespace(&mut self.buf)
                        .ok_or(CueFormatParseErr::ExpectedFileType)?;

                    self.file_builder = Some(CueFileBuilder {
                        has_pregap: false,
                        filename:   filename.to_owned(),
                        tracks:     Vec::new(),
                        bytes_len:  0,
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

                    // data pregap
                    if !self.files.is_empty() && track_kind == TrackKind::Mode2_2352 {
                        self.track_lba += 150;
                    }

                    let start = self.indices.len() as u8;
                    self.track_builder = Some(Track {
                        number,
                        kind: track_kind,
                        indices: start..start,
                        abs_lba: self.file_lba + self.track_lba,
                        rel_lba: self.track_lba,
                        index_0_pregap: false,
                    });

                    // data postgap
                    if track_kind == TrackKind::Mode2_2352 {
                        self.track_lba += 150;
                    }

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
                    tb.indices.end += 1;
                    if number == 0 {
                        tb.index_0_pregap = true;
                    }

                    if number != 1 || (!tb.index_0_pregap && number != 0) {
                        self.track_lba += mss_to_lba(timestamp);
                    }

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
        debug_assert!(idx != 0, "index cannot be 0");

        let track_idx = self.tracklist[usize::from(track - 1)];
        let track = &self.tracks[usize::from(track_idx)];

        let start = track.indices.start;
        let end = track.indices.end;

        let indices = &self.indices[usize::from(idx + start - 1)..usize::from(idx + end - 1)];
        indices
            .iter()
            .find(|index| index.number == idx)
            .expect("todo: index not found")
    }

    pub fn lba_to_byte(&self, lba: u32) -> (FileIdx, u64) {
        let (file_idx, relative_lba) = self.relative_lba(lba);

        let byte = CdromCursor::new(relative_lba, 0).to_byte();

        tracing::info!(?lba, ?relative_lba, ?file_idx);

        (file_idx, byte)
    }

    pub fn relative_lba(&self, lba: u32) -> (FileIdx, u32) {
        let file_idx = self
            .files
            .iter()
            .position(|file| file.lba > lba)
            .map_or(self.files.len().saturating_sub(1), |idx| {
                idx.saturating_sub(1)
            });

        let file = &self.files[file_idx];
        let relative_lba = lba - file.lba;
        let relative_lba = relative_lba.saturating_sub(150);

        (file_idx, relative_lba)
    }
}

#[cfg(test)]
mod cuetests {
    use crate::cdrom::cdrom_cue_parser::{CueParser, CueParserStep, File};
    use crate::cdrom::cdrom_format::SECTOR_TOTAL_SIZE;

    struct MockFile;

    impl File for MockFile {
        fn size(&self) -> u64 {
            32 * SECTOR_TOTAL_SIZE as u64
        }
    }

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
