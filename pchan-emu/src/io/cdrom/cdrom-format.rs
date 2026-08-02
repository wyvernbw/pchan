use std::marker::Destruct;
use std::str::FromStr;

use arbitrary_int::prelude::*;
use bitbybit::*;

use crate::io::cdrom::cdrom_cmds::SetModeSectSize;

#[bitfield(u8, debug)]
pub struct Bcd {
    /// the least significant digit
    #[bits(0..=3, rw)]
    digit_01: u4,
    /// the most significant digit
    #[bits(4..=7, rw)]
    digit_02: u4,
}

impl std::fmt::Display for Bcd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02}", self.unpack())
    }
}

impl Bcd {
    pub const fn unpack(self) -> u8 {
        self.digit_02().value() * 10 + self.digit_01().value()
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
#[derive(Debug, Clone, Copy, derive_more::Display)]
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
pub struct CueFormat {
    pub filename:   String,
    pub index_list: Vec<CueIndex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueIndex {
    pub num:    u8,
    pub min:    u8,
    pub second: u8,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum CueFormatParseErr {
    #[error("expected `FILE` attribute")]
    ExpectedFile,
    #[error("expected filename in double quotes")]
    ExpectedQuotes,
    #[error("`INDEX` attribute must be followed by a number")]
    MissingIndexNumber,
    #[error("`INDEX` number must be followed by a timestamp")]
    MissingTimestamp,
}

impl FromStr for CueFormat {
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
                let quote_end = s.find(r#"""#)?;
                Some(&s[..quote_end])
            })
            .ok_or(CueFormatParseErr::ExpectedQuotes)?;

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
                    .map(|time| time.split(":").take(2))
                    .and_then(|mut split| split.next().zip(split.next()))
                    .ok_or(CueFormatParseErr::MissingTimestamp)?;
                let min = min
                    .parse::<u8>()
                    .map_err(|_| CueFormatParseErr::MissingTimestamp)?;
                let sec = sec
                    .parse::<u8>()
                    .map_err(|_| CueFormatParseErr::MissingTimestamp)?;
                Ok(CueIndex {
                    num,
                    min,
                    second: sec,
                })
            })
            .collect::<Result<Vec<_>, CueFormatParseErr>>()?;

        Ok(CueFormat {
            filename: filename.to_owned(),
            index_list,
        })
    }
}

#[cfg(test)]
mod cuetests {
    use crate::io::cdrom::cdrom_format::{CueFormat, CueFormatParseErr, CueIndex};

    #[test]
    fn test_basic_cue() {
        let string = r#"
            FILE "Valkyrie Profile (USA) (Disc 1).bin" BINARY
              TRACK 01 MODE2/2352
                INDEX 01 00:00:00
            "#;
        assert_eq!(
            string.parse::<CueFormat>(),
            Ok(CueFormat {
                filename:   "Valkyrie Profile (USA) (Disc 1).bin".to_string(),
                index_list: vec![CueIndex {
                    num:    1,
                    min:    00,
                    second: 00,
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
            "#;
        assert_eq!(
            string.parse::<CueFormat>(),
            Ok(CueFormat {
                filename:   "Valkyrie Profile (USA) (Disc 1).bin".to_string(),
                index_list: vec![CueIndex {
                    num:    1,
                    min:    00,
                    second: 2,
                }],
            })
        );
    }
}
