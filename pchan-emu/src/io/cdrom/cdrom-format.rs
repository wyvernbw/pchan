use std::marker::Destruct;
use std::str::FromStr;

use arbitrary_int::prelude::*;
use bitbybit::*;

use crate::io::cdrom::cdrom_cmds::SetModeSectSize;

#[bitfield(u8)]
pub struct Bcd {
    /// the least significant digit
    #[bits(0..=3, rw)]
    digit_01: u4,
    /// the most significant digit
    #[bits(4..=7, rw)]
    digit_02: u4,
}

impl Bcd {
    pub const fn unpack(self) -> u8 {
        self.digit_02().value() * 10 + self.digit_01().value()
    }
}

impl const From<Bcd> for u8 {
    fn from(val: Bcd) -> Self {
        val.unpack()
    }
}

impl const From<u8> for Bcd {
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
#[derive(Debug, Clone, Copy)]
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

impl CdromCursor {
    pub const fn from_mss<T: [const] Into<u8> + [const] Destruct>(mss: Mss<T>) -> Self {
        let min = mss.min.into() as u32;
        let sec = mss.sec.into() as u32;
        let sect = mss.sect.into() as u32;
        Self {
            lba:  min * (60 * 75) + sec * 75 + sect,
            byte: 0,
        }
    }

    pub const fn lba_to_bytes(&self) -> u32 {
        self.lba * 0x924
    }

    pub const fn advance_by(&mut self, mut by_bytes: u32, sect_size: SetModeSectSize) {
        match sect_size {
            SetModeSectSize::DataOnly0x800 => {
                let pad = 0x924 - 0x800;
                let mut to_end = 0x924 - self.byte;
                while by_bytes > to_end {
                    by_bytes -= to_end;
                    self.byte = pad;
                    self.lba += 1;
                    to_end = 0x924;
                }
                self.byte += by_bytes
            }
            SetModeSectSize::Whole0x924 => {
                let mut to_end = 0x924 - self.byte;
                while by_bytes > to_end {
                    by_bytes -= to_end;
                    self.byte = 0x0;
                    self.lba += 1;
                    to_end = 0x924;
                }
                self.byte += by_bytes
            }
        };
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
