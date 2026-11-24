use core::fmt;
use std::marker::Destruct;

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

pub const SECTOR_TOTAL_SIZE: usize = 0x930;

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
        u64::from(self.lba) * SECTOR_TOTAL_SIZE as u64 + u64::from(self.byte)
    }

    pub const fn advance_by(&mut self, by_sectors: u32, by_bytes: u32) {
        self.lba += by_sectors;
        self.byte += by_bytes;
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
