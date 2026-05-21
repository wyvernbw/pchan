use std::marker::Destruct;

use arbitrary_int::prelude::*;
use bitbybit::*;

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

#[derive(Default, derive_more::Debug, Clone)]
pub struct CdromCursor {
    lba: u32,
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
            lba: min * (60 * 75) + sec * 75 + sect,
        }
    }
}
