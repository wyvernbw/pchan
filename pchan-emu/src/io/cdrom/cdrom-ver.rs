use derive_more as d;
use pchan_encode::derive::Endecode;
use pchan_utils::hex;

#[derive(d::Deref, d::DerefMut, Clone, Copy, d::Debug, Endecode)]
#[debug("{}", hex(self.0))]
pub struct CDRomVer([u8; 4]);

/// 94h,09h,19h,C0h  ;PSX (PU-7)               19 Sep 1994, version vC0 (a)
pub static PSXPU7: CDRomVer = CDRomVer([0x94, 0x09, 0x19, 0xc0]);
/// 95h,07h,24h,C1h  ;PSX (LATE-PU-8)          24 Jul 1995, version vC1 (b)
pub static PSXLATEPU8: CDRomVer = CDRomVer([0x95, 0x07, 0x24, 0xc1]);

impl Default for CDRomVer {
    fn default() -> Self {
        PSXLATEPU8
    }
}
