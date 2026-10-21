use core::alloc::Allocator;
use core::fmt;
use core::str::FromStr;
use std::marker::Destruct;

use arbitrary_int::prelude::*;
use bitbybit::bitfield;
use pchan_encode::derive::Endecode;
use pchan_encode::{Endecode, EndecodeError, Stream};

#[bitfield(u8, debug)]
#[derive(Endecode, Default)]
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

#[derive(Default, derive_more::Debug, Clone, Copy, Endecode)]
pub struct CdromCursor {
    pub lba:  u32,
    pub byte: u32,
}

/// (minute, second, sector) tuple
#[derive(Debug, Clone, Copy, derive_more::Display, Default, Endecode)]
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

impl CdromCursor {
    pub const fn from_mss<T: [const] Into<u8> + [const] Destruct>(mss: Mss<T>) -> Self {
        let min = mss.min.into() as u32;
        let sec = mss.sec.into() as u32;
        let sect = mss.sect.into() as u32;
        Self {
            lba:  (min * (60 * 75) + sec * 75 + sect).saturating_sub(150),
            byte: 0,
        }
    }

    pub fn to_mss<T: From<u8>>(self) -> Mss<T> {
        let lba = self.lba + 150;

        let sect = (lba % 75) as u8;
        let sec = (lba / 75 % 60) as u8;
        let min = (lba / 75 / 60) as u8;

        Mss {
            min:  T::from(min),
            sec:  T::from(sec),
            sect: T::from(sect),
        }
    }

    pub const fn to_byte(self) -> u32 {
        self.lba * SECTOR_USER_SIZE as u32 + self.byte
    }

    pub const fn advance_by(&mut self, by_sectors: u32, by_bytes: u32) {
        self.lba += by_sectors;
        self.byte += by_bytes;
    }
}

/// TODO: add support for multiple tracks
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueSheet {
    pub filename: String,
    pub tracks:   Vec<Track>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    pub number:     u8,
    pub index_list: Vec<CueIndex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueIndex {
    pub sect:   u8,
    pub min:    u8,
    pub second: u8,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum CueFormatParseErr {
    #[error("expected `FILE` attribute")]
    ExpectedFile,
    #[error("expected filename in double quotes")]
    ExpectedQuotes,
    #[error("expected track number for TRACK")]
    ExpectedTrackNumber,
    #[error("`INDEX` attribute must be followed by a number")]
    MissingIndexNumber,
    #[error("`INDEX` number must be followed by a timestamp")]
    MissingTimestamp,
}

impl FromStr for CueSheet {
    type Err = CueFormatParseErr;

    /// parse cue format
    /// ```cue
    /// FILE "test.bin" BINARY
    ///   TRACK 01 MODE2/2352
    ///     INDEX 01 00:00:00
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        // FILE
        let s = match s.strip_prefix("FILE") {
            Some(rest) => Ok(rest),
            None => Err(CueFormatParseErr::ExpectedFile),
        }?;
        let s = s.trim();

        let filename = s
            .strip_prefix(r#"""#)
            .and_then(|s| {
                let quote_end = s.find('"')?;
                Some(&s[..quote_end])
            })
            .ok_or(CueFormatParseErr::ExpectedQuotes)?;

        let track_list = s
            .split("TRACK ")
            .skip(1)
            .map(|s| {
                let Some(num) = s.split_whitespace().next() else {
                    return Err(CueFormatParseErr::ExpectedTrackNumber);
                };

                let num = num
                    .parse::<u8>()
                    .map_err(|_| CueFormatParseErr::ExpectedTrackNumber)?;

                let index_list = s
                    .split("INDEX ")
                    .skip(1)
                    .map(|s| {
                        let mut index = s.split_whitespace().take(2);
                        let num = index
                            .next()
                            .and_then(|num| num.parse::<u8>().ok())
                            .ok_or(CueFormatParseErr::MissingIndexNumber)?;
                        let (min, sec) = index
                            .next()
                            .map(|time| time.split(':').take(2))
                            .and_then(|mut split| split.next().zip(split.next()))
                            .ok_or(CueFormatParseErr::MissingTimestamp)?;
                        let min = min
                            .parse::<u8>()
                            .map_err(|_| CueFormatParseErr::MissingTimestamp)?;
                        let sec = sec
                            .parse::<u8>()
                            .map_err(|_| CueFormatParseErr::MissingTimestamp)?;
                        Ok(CueIndex {
                            sect: num,
                            min,
                            second: sec,
                        })
                    })
                    .collect::<Result<Vec<_>, CueFormatParseErr>>()?;
                Ok(Track {
                    number: num,
                    index_list,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(CueSheet {
            filename: filename.to_owned(),
            tracks:   track_list,
        })
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
#[derive(Endecode)]
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

impl<A: Allocator + Clone> Endecode<A> for SyncHeader {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        let mut buf = [0u8; SYNC_HEADER_LEN];
        buf.endecode(sink, alloc)
    }

    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(SYNC_HEADER.len()))
    }
}

/// `4 Header (Minute,Second,Sector,Mode)`
#[derive(Endecode)]
struct Header {
    mss:  Mss<Bcd>,
    mode: HeaderMode,
}

#[derive(Debug, Clone, Copy, Endecode)]
enum HeaderMode {
    Mode2 = 0x02,
}

/// `4 Sub-Header (File, Channel, Submode AND DFh, Codinginfo)`
#[derive(Endecode)]
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
#[derive(Endecode)]
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
/// ```plaintext
///   0-7 Reserved (00h)
/// ```
/// When used for XA-ADPCM audio sectors:
/// ```plaintext
///   0-1 Mono/Stereo     (0=Mono, 1=Stereo, 2-3=Reserved)
///   2-2 Sample Rate     (0=37800Hz, 1=18900Hz, 2-3=Reserved)
///   4-5 Bits per Sample (0=Normal/4bit, 1=8bit, 2-3=Reserved)
///   6   Emphasis        (0=Normal/Off, 1=Emphasis)
///   7   Reserved        (0)
/// ```
#[bitfield(u8, default = 0x00)]
#[derive(Endecode)]
struct CodingInfo {
    #[bits(0..=1, rw)]
    mono_stereo: u2,
    #[bits(2..=3)]
    sample_rate: XAADPCMSampleRate,
    // TODO: the rest of the fields
}

/// # 1st Subheader byte - File Number (FN)
#[derive(Endecode)]
struct FileNumber(u8);
/// # 2nd Subheader byte - Channel Number (CN)
#[derive(Endecode)]
struct ChannelNumber(u8);

#[cfg(test)]
mod cuetests {
    use crate::io::cdrom::cdrom_format::{CueIndex, CueSheet, Track};

    #[test]
    fn test_basic_cue() {
        let string = r#"
            FILE "Valkyrie Profile (USA) (Disc 1).bin" BINARY
              TRACK 01 MODE2/2352
                INDEX 01 00:00:00
            "#;
        assert_eq!(
            string.parse::<CueSheet>(),
            Ok(CueSheet {
                filename: "Valkyrie Profile (USA) (Disc 1).bin".to_string(),
                tracks:   vec![Track {
                    number:     1,
                    index_list: vec![CueIndex {
                        sect:   1,
                        min:    00,
                        second: 00,
                    }],
                }],
            })
        );
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
        assert_eq!(
            string.parse::<CueSheet>(),
            Ok(CueSheet {
                filename: "Valkyrie Profile (USA) (Disc 1).bin".to_string(),
                tracks:   vec![
                    Track {
                        number:     1,
                        index_list: vec![CueIndex {
                            sect:   1,
                            min:    00,
                            second: 2,
                        }],
                    },
                    Track {
                        number:     2,
                        index_list: vec![CueIndex {
                            sect:   1,
                            min:    00,
                            second: 4,
                        }],
                    }
                ],
            })
        );
    }
}
