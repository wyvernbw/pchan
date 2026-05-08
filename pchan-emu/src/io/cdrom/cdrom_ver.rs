use derive_more as d;
pub type CDRomVer = [u8; 4];

/// 94h,09h,19h,C0h  ;PSX (PU-7)               19 Sep 1994, version vC0 (a)
pub static PSXPU7: CDRomVer = [0x94, 0x09, 0x19, 0xc0];

#[derive(Debug, Clone, Copy, d::Deref)]
pub struct CDRomVerPtr(&'static CDRomVer);

impl Default for CDRomVerPtr {
    fn default() -> Self {
        Self(&PSXPU7)
    }
}

impl CDRomVerPtr {
    pub fn iter(self) -> impl Iterator<Item = u8> {
        self.0.iter().copied()
    }
}
