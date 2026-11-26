use core::fmt;
use core::mem::{MaybeUninit, transmute};
use std::marker::Destruct;

use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};

use crate::spu::{NEG_ADPCM_TABLE, POS_ADPCM_TABLE};

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

#[derive(Default, derive_more::Debug, derive_more::Display, Clone, Copy)]
#[display("{}", &self.to_mss::<u8>())]
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
/// ```
#[derive(Default, Debug, Clone)]
pub struct DataSectorHeader {
    pub sync:   SyncHeader,
    pub header: Header,
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
#[derive(Default, Debug, Clone)]
pub struct Mode2SectorMetadata {
    pub common:    DataSectorHeader,
    pub subheader: Subheader,
}

#[derive(Default, Debug, Clone)]
struct SyncHeader;

static SYNC_HEADER: &[u8] = &[
    0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
];
const SYNC_HEADER_LEN: usize = SYNC_HEADER.len();

/// `4 Header (Minute,Second,Sector,Mode)`
#[derive(Default, Debug, Clone)]
pub struct Header {
    pub mss:  Mss<Bcd>,
    pub mode: SectorMode,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum SectorMode {
    #[default]
    Mode2 = 0x02,
}

/// `4 Sub-Header (File, Channel, Submode AND DFh, Codinginfo)`
#[derive(Debug, Clone, Default)]
pub struct Subheader {
    file:        FileNumber,
    channel:     ChannelNumber,
    pub submode: Submode,
    pub cinfo:   CodingInfo,
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
#[bitfield(u8, debug)]
#[derive(Default)]
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

impl Submode {
    pub fn realtime_audio(&self) -> bool {
        self.rt() && self.audio()
    }
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
#[bitfield(u8, default = 0x00, debug)]
pub struct CodingInfo {
    #[bit(0, rw)]
    mono_stereo: XaAdpcmChannelFormat,
    #[bit(2, rw)]
    sample_rate: XaAdpcmSampleRate,
    #[bit(4, rw)]
    bps:         XaAdpcmBitsPerSample,
    #[bit(6)]
    emphasis:    bool,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum XaAdpcmChannelFormat {
    Mono,
    Stereo,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum XaAdpcmSampleRate {
    Normal,
    Double,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum XaAdpcmBitsPerSample {
    _4Bit,
    _8Bit,
}

/// # 1st Subheader byte - File Number (FN)
#[derive(Debug, Clone, Default)]
struct FileNumber(u8);
/// # 2nd Subheader byte - Channel Number (CN)
#[derive(Debug, Clone, Default)]
struct ChannelNumber(u8);

/// # XA-ADPCM Header Bytes
/// ```plaintext
///   0-3   Shift  (0..12) (0=Loudest) (13..15=Reserved/Same as 9)
///   4-5   Filter (0..3) (only four filters, unlike SPU-ADPCM which has five)
///   6-7   Unused (should be 0)
/// ```
#[bitfield(u8, default = 0x00)]
pub struct XaAdpcmBlockHeader {
    #[bits(0..=3, rw)]
    shift:  u4,
    #[bits(4..=5, rw)]
    filter: u2,
}

pub struct XaAdpcmSectionHeader([XaAdpcmBlockHeader; 8]);

pub const XA_ADPCM_SECTION_DATA: usize = 4 * 28;

pub struct XaAdpcmSection<'buf> {
    headers: XaAdpcmSectionHeader,
    data:    &'buf [u8; XA_ADPCM_SECTION_DATA],
}

pub struct XaAdpcmSectorData<'buf> {
    sections: [XaAdpcmSection<'buf>; 0x12],
}

pub struct XaAdpcmSector<'buf> {
    pub meta: Mode2SectorMetadata,
    pub data: XaAdpcmSectorData<'buf>,
}

#[derive(thiserror::Error, Debug)]
pub enum XaAdpcmSectorDecodeError {
    #[error("xa-adpcm buffer underflow during decode.")]
    Underflow(#[from] CursorErr),
}

pub struct ByteCursor<'a> {
    buf: &'a [u8],
}

#[derive(thiserror::Error, Debug)]
#[error("cursor overflowed buffer.")]
pub struct CursorErr;

impl<'a> ByteCursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    fn read(&mut self, len: usize) -> Result<&[u8], CursorErr> {
        let (head, rest) = self.buf.split_at_checked(len).ok_or(CursorErr)?;
        self.buf = rest;
        Ok(head)
    }
    fn read_chunk<const N: usize>(&mut self) -> Result<&'a [u8; N], CursorErr> {
        let (head, rest) = self.buf.split_first_chunk::<N>().ok_or(CursorErr)?;
        self.buf = rest;
        Ok(head)
    }
    fn advance(&mut self, by: usize) -> Result<(), CursorErr> {
        let _ = self.read(by)?;
        Ok(())
    }
}

impl<'buf> XaAdpcmSector<'buf> {
    pub fn new(buf: &'buf [u8]) -> Result<Self, XaAdpcmSectorDecodeError> {
        Self::decode(&mut ByteCursor::new(buf))
    }
    pub fn decode(buf: &mut ByteCursor<'buf>) -> Result<Self, XaAdpcmSectorDecodeError> {
        Ok(Self {
            meta: Mode2SectorMetadata::decode(buf)?,
            data: XaAdpcmSectorData::decode(buf)?,
        })
    }
}

impl DataSectorHeader {
    pub fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        Ok(Self {
            sync:   SyncHeader::decode(buf)?,
            header: Header::decode(buf)?,
        })
    }
}

impl Mode2SectorMetadata {
    pub fn decode_partial(
        header: DataSectorHeader,
        buf: &mut ByteCursor,
    ) -> Result<Self, CursorErr> {
        let meta = Self {
            common:    header,
            subheader: Subheader::decode(buf)?,
        };
        // copy of subheader
        let _ = Subheader::decode(buf)?;

        Ok(meta)
    }
    pub fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        let common = DataSectorHeader::decode(buf)?;
        Self::decode_partial(common, buf)
    }
}

impl SyncHeader {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        buf.advance(SYNC_HEADER_LEN)?;
        Ok(SyncHeader)
    }
}

impl Header {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        let mss = buf.read_chunk::<3>()?;
        let mss = Mss::new(
            Bcd::new_with_raw_value(mss[0]),
            Bcd::new_with_raw_value(mss[1]),
            Bcd::new_with_raw_value(mss[2]),
        );

        let mode = buf.read_chunk::<1>()?[0];
        let mode = match mode {
            2 => SectorMode::Mode2,
            _ => SectorMode::Mode2,
        };
        Ok(Self { mss, mode })
    }
}

impl Subheader {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        Ok(Self {
            file:    FileNumber::decode(buf)?,
            channel: ChannelNumber::decode(buf)?,
            submode: Submode::decode(buf)?,
            cinfo:   CodingInfo::decode(buf)?,
        })
    }
}

impl FileNumber {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        let number = buf.read(1)?[0];
        Ok(Self(number))
    }
}

impl ChannelNumber {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        let number = buf.read(1)?[0];
        Ok(Self(number))
    }
}

impl Submode {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        let number = buf.read(1)?[0];
        Ok(Self::new_with_raw_value(number))
    }
}

impl CodingInfo {
    fn decode(buf: &mut ByteCursor) -> Result<Self, CursorErr> {
        let number = buf.read(1)?[0];
        Ok(Self::new_with_raw_value(number))
    }
}

impl<'buf> XaAdpcmSectorData<'buf> {
    pub fn decode(buf: &mut ByteCursor<'buf>) -> Result<Self, XaAdpcmSectorDecodeError> {
        let mut sections = const { [const { MaybeUninit::<XaAdpcmSection>::uninit() }; 0x12] };
        for section in &mut sections {
            section.write(XaAdpcmSection::decode(buf)?);
        }

        // SAFETY: `MaybeUninit` has the same layout, and all values are initialized above
        let sections = unsafe { transmute::<[MaybeUninit<_>; _], [XaAdpcmSection; _]>(sections) };

        Ok(Self { sections })
    }
}

impl<'buf> XaAdpcmSection<'buf> {
    pub fn decode(buf: &mut ByteCursor<'buf>) -> Result<Self, XaAdpcmSectorDecodeError> {
        let headers = XaAdpcmSectionHeader::decode(buf)?;
        let bytes = buf.read_chunk::<XA_ADPCM_SECTION_DATA>()?;

        Ok(Self {
            headers,
            data: bytes,
        })
    }
}

impl XaAdpcmSectionHeader {
    fn decode(buf: &mut ByteCursor) -> Result<Self, XaAdpcmSectorDecodeError> {
        buf.advance(4)?;

        let header = buf.read_chunk::<8>()?;
        let (_, header, _) = unsafe {
            const {
                assert!(align_of::<XaAdpcmBlockHeader>() == align_of::<u8>());
            };
            header.align_to::<XaAdpcmBlockHeader>()
        };
        let header: &[XaAdpcmBlockHeader; 8] = header.try_into().expect("unreachable");
        let header = *header;

        buf.advance(4)?;

        Ok(XaAdpcmSectionHeader(header))
    }
}

#[derive(Default)]
pub struct XaAdpcmDecoder {
    left:  [i16; 3],
    right: [i16; 3],
}

impl XaAdpcmDecoder {
    pub fn decode_xaadpcm_sector(&mut self, sector: &XaAdpcmSector, dest: &mut [i16]) {
        let channel_format = sector.meta.subheader.cinfo.mono_stereo();

        for section in &sector.data.sections {
            for blk in 0..4 {
                let header_1 = section.headers.0[blk * 2];
                let header_2 = section.headers.0[blk * 2 + 1];
                match channel_format {
                    XaAdpcmChannelFormat::Mono => {
                        Self::decode_xaadpcm_block(
                            section.data,
                            dest,
                            blk,
                            0,
                            header_1,
                            &mut self.left,
                        );
                        Self::decode_xaadpcm_block(
                            section.data,
                            dest,
                            blk,
                            1,
                            header_2,
                            &mut self.left,
                        );
                    }
                    XaAdpcmChannelFormat::Stereo => {
                        Self::decode_xaadpcm_block(
                            section.data,
                            dest,
                            blk,
                            0,
                            header_1,
                            &mut self.left,
                        );
                        Self::decode_xaadpcm_block(
                            section.data,
                            dest,
                            blk,
                            1,
                            header_2,
                            &mut self.right,
                        );
                    }
                }
            }
        }
    }

    fn decode_xaadpcm_block(
        src: &[u8; XA_ADPCM_SECTION_DATA],
        dest: &mut [i16],
        block: usize,
        nibble: usize,
        header: XaAdpcmBlockHeader,
        queue: &mut [i16; 3],
    ) {
        let shift = header.shift().value();
        let shift = if shift > 12 { 9 } else { shift };
        let shift = 12 - shift;

        let filter = header.filter().value();

        // SAFETY: filter is between 0x0 and 0x4
        let (f0, f1) = unsafe {
            let f0 = POS_ADPCM_TABLE.get_unchecked(filter as usize);
            let f1 = NEG_ADPCM_TABLE.get_unchecked(filter as usize);
            (f0, f1)
        };

        for (idx, dest) in dest.iter_mut().skip(block * 28).take(28).enumerate() {
            let sample = src[block + idx * 4];
            let sample = sample >> ((4 * nibble) & 0x0F);
            let sample = i32::from((sample as i8) << 4 >> 4); // sign extend from bit 3

            {
                let s1 = i32::from(queue[1]);
                let s2 = i32::from(queue[2]);

                let sample = (sample << shift) + ((s1 * f0 + s2 * f1 + 32) / 64);
                let sample = sample.clamp(-0x8000, 0x7fff);
                *dest = sample as i16;
                queue[0] = *dest;
            }

            queue[2] = queue[1];
            queue[1] = queue[0];
        }
    }
}
