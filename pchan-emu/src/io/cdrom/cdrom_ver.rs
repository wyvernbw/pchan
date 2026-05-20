use derive_more as d;
pub type CDRomVer = [u8; 4];

/// 94h,09h,19h,C0h  ;PSX (PU-7)               19 Sep 1994, version vC0 (a)
pub static PSXPU7: CDRomVer = [0x94, 0x09, 0x19, 0xc0];
/// 95h,07h,24h,C1h  ;PSX (LATE-PU-8)          24 Jul 1995, version vC1 (b)
pub static PSXLATEPU8: CDRomVer = [0x95, 0x07, 0x24, 0xc1];

#[derive(Debug, Clone, Copy, d::Deref)]
pub struct CDRomVerPtr(&'static CDRomVer);

impl Default for CDRomVerPtr {
    fn default() -> Self {
        Self(&PSXLATEPU8)
    }
}

impl CDRomVerPtr {
    pub fn iter(self) -> impl Iterator<Item = u8> {
        self.0.iter().copied()
    }

    pub fn to_owned(&self) -> CDRomVer {
        *self.0
    }

    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}
