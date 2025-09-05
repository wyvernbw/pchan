use std::ops::{Add, Mul};

use thiserror::Error;
use tracing::instrument;

pub const fn kb(value: usize) -> usize {
    value * 1024
}

pub fn buffer(size: usize) -> Box<[u8]> {
    vec![0u8; size].into_boxed_slice()
}

const MEM_SIZE: usize = kb(2048) + kb(8192) + kb(1) + kb(8) + kb(8) + kb(2048) + kb(512) + 512;

pub struct Memory(Box<[u8]>);

impl AsRef<[u8]> for Memory {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Memory {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl Default for Memory {
    fn default() -> Self {
        Memory(buffer(MEM_SIZE))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PhysAddr(pub u32);

impl PhysAddr {
    pub const fn to_kseg0(self) -> KSEG0Addr {
        KSEG0Addr(self.0 + 0x8000_0000u32)
    }
    pub const fn to_kseg1(self) -> KSEG1Addr {
        KSEG1Addr(self.0 + 0xA000_0000u32)
    }
}

impl core::fmt::Debug for PhysAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PhysAddr")
            .field_with(|f| write!(f, "0x{:08X}", self.0))
            .finish()
    }
}

#[derive(Debug, Error)]
#[error("address header does not match KSEG0 or KSEG1 headers")]
pub struct PhysAddrWrongHeader;

impl PhysAddr {
    #[instrument(err, fields(address = %format!("0x{:08X}", address)))]
    pub fn try_map(address: u32) -> Result<Self, PhysAddrWrongHeader> {
        let header = address & 0xE000_0000;
        match header {
            // KSEG0
            0x8000_0000 => Ok(KSEG0Addr(address).to_phys()),
            // KSEG1
            0xA000_0000 => Ok(KSEG1Addr(address).to_phys()),
            _ => Err(PhysAddrWrongHeader),
        }
    }
    pub fn map(address: u32) -> Self {
        Self::try_map(address).unwrap()
    }
    pub fn new(address: u32) -> Self {
        PhysAddr(address)
    }
    pub fn as_usize(self) -> usize {
        self.into()
    }
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl Add<u32> for PhysAddr {
    type Output = PhysAddr;

    fn add(self, rhs: u32) -> Self::Output {
        PhysAddr(self.0 + rhs)
    }
}

#[derive(derive_more::Debug, Clone, Copy, PartialEq, Eq, derive_more::Add)]
#[debug("0x{:08X}", self.0)]
pub struct KSEG0Addr(pub u32);
#[derive(derive_more::Debug, Clone, Copy, PartialEq, Eq, derive_more::Add)]
#[debug("0x{:08X}", self.0)]
pub struct KSEG1Addr(pub u32);
#[derive(derive_more::Debug, Clone, Copy, PartialEq, Eq)]
#[debug("0x{:08X}", self.0)]
pub struct Addr(pub u32);

impl KSEG0Addr {
    pub const fn to_phys(self) -> PhysAddr {
        PhysAddr(self.0 - 0x8000_0000u32)
    }
    pub const fn from_phys(value: u32) -> Self {
        PhysAddr(value).to_kseg0()
    }
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl const Add<u32> for KSEG0Addr {
    type Output = KSEG0Addr;

    fn add(self, rhs: u32) -> Self::Output {
        KSEG0Addr(self.0 + rhs)
    }
}

impl KSEG1Addr {
    pub const fn to_phys(self) -> PhysAddr {
        PhysAddr(self.0 - 0xA000_0000u32)
    }
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<KSEG0Addr> for PhysAddr {
    fn from(value: KSEG0Addr) -> Self {
        value.to_phys()
    }
}

impl const From<KSEG0Addr> for Addr {
    fn from(value: KSEG0Addr) -> Self {
        Addr(value.0)
    }
}

impl From<KSEG1Addr> for PhysAddr {
    fn from(value: KSEG1Addr) -> Self {
        value.to_phys()
    }
}

impl const From<KSEG1Addr> for Addr {
    fn from(value: KSEG1Addr) -> Self {
        Addr(value.0)
    }
}

impl const Add<u32> for KSEG1Addr {
    type Output = KSEG1Addr;

    fn add(self, rhs: u32) -> Self::Output {
        KSEG1Addr(self.0 + rhs)
    }
}

impl TryFrom<Addr> for PhysAddr {
    type Error = PhysAddrWrongHeader;

    fn try_from(value: Addr) -> Result<Self, Self::Error> {
        PhysAddr::try_map(value.0)
    }
}

impl From<PhysAddr> for usize {
    fn from(value: PhysAddr) -> Self {
        value.0 as usize
    }
}

impl const Mul<u32> for Addr {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        Addr(self.0 * rhs)
    }
}

#[cfg(test)]
mod physaddr_tests {
    use super::PhysAddr;

    #[test]
    fn kseg0_to_physical() {
        let addr = PhysAddr::map(0x8000_0000);
        assert_eq!(addr.0, 0x0000_0000);

        let addr = PhysAddr::map(0x8020_0100);
        assert_eq!(addr.0, 0x0020_0100);
    }

    #[test]
    fn kseg1_to_physical() {
        let addr = PhysAddr::map(0xA000_0000);
        assert_eq!(addr.0, 0x0000_0000);

        let addr = PhysAddr::map(0xA020_0100);
        assert_eq!(addr.0, 0x0020_0100);
    }

    #[test]
    #[should_panic]
    fn unmapped_address_panics() {
        PhysAddr::map(0x0000_0000); // KUSEG (unmapped)
    }

    #[test]
    fn try_new_returns_none_for_unmapped() {
        assert!(PhysAddr::try_map(0x0000_0000).is_err());
        assert!(PhysAddr::try_map(0xC000_0000).is_err()); // KSEG2
    }
}

pub trait MapAddress {
    fn map(self) -> PhysAddr;
}

impl MapAddress for u32 {
    fn map(self) -> PhysAddr {
        PhysAddr::map(self)
    }
}

impl MapAddress for Addr {
    fn map(self) -> PhysAddr {
        self.0.map()
    }
}
