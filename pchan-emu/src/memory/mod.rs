use std::ops::Add;

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

impl Default for Memory {
    fn default() -> Self {
        Memory(buffer(MEM_SIZE))
    }
}

// cost of dynamic dispatch doesnt matter since we are
// on the cold path anyways
type PrintableAddress = Box<dyn core::fmt::Debug>;

#[derive(Error, Debug)]
pub enum MemReadError {
    #[error("read from unmapped address 0x{0:08X?}")]
    UnmappedRead(PrintableAddress),
    #[error(transparent)]
    DerefErr(DerefError),
    #[error("out of bounds read at address 0x{0:08X}")]
    OutOfBoundsRead(u32),
}

impl MemReadError {
    fn unmapped(addr: impl core::fmt::Debug + 'static) -> Self {
        MemReadError::UnmappedRead(Box::new(addr) as Box<_>)
    }
}

#[derive(Error, Debug)]
pub enum MemWriteError {
    #[error("write to unmapped address {0:?}")]
    UnmappedWrite(PrintableAddress),
    #[error(transparent)]
    DerefErr(DerefError),
    #[error("partial write into buffer {0:?} (size: {1}) from buffer {2:?} (size:{3})")]
    MismatchedBuffers(*const u8, usize, *const u8, usize),
    #[error("out of bounds write at address {0}")]
    OutOfBoundsWrite(u32),
}

impl MemWriteError {
    fn unmapped(addr: impl core::fmt::Debug + 'static) -> Self {
        MemWriteError::UnmappedWrite(Box::new(addr) as Box<_>)
    }
}

#[derive(Debug, Error)]
#[error("error dereferencing slice")]
pub struct DerefError;

pub(crate) trait MemRead: Sized {
    fn from_slice(buf: &[u8]) -> Result<Self, DerefError>;
}

impl MemRead for u8 {
    #[inline]
    fn from_slice(buf: &[u8]) -> Result<u8, DerefError> {
        Ok(buf[0])
    }
}

impl MemRead for u16 {
    fn from_slice(buf: &[u8]) -> Result<u16, DerefError> {
        let buf = buf.as_array().ok_or(DerefError)?;
        Ok(u16::from_be_bytes(*buf))
    }
}

impl MemRead for u32 {
    fn from_slice(buf: &[u8]) -> Result<u32, DerefError> {
        let buf = buf.as_array().ok_or(DerefError)?;
        Ok(u32::from_be_bytes(*buf))
    }
}

pub(crate) trait MemWrite<const N: usize = { size_of::<Self>() }>: Sized {
    fn to_bytes(&self) -> [u8; N];
    fn write(buf: &mut [u8], value: &Self) -> Result<(), MemWriteError>
    where
        [(); N]:,
    {
        let bytes = &value.to_bytes();
        if bytes.len() != buf.len() {
            return Err(MemWriteError::MismatchedBuffers(
                buf.as_ptr(),
                buf.len(),
                bytes.as_ptr(),
                bytes.len(),
            ));
        };
        buf.copy_from_slice(bytes);
        Ok(())
    }
}

impl MemWrite for u8 {
    #[inline]
    fn to_bytes(&self) -> [u8; 1] {
        [*self]
    }
    fn write(buf: &mut [u8], value: &Self) -> Result<(), MemWriteError> {
        buf[0] = *value;
        Ok(())
    }
}

impl MemWrite for u16 {
    #[inline]
    fn to_bytes(&self) -> [u8; 2] {
        self.to_be_bytes()
    }
}

impl MemWrite for u32 {
    #[inline]
    fn to_bytes(&self) -> [u8; 4] {
        self.to_be_bytes()
    }
}

pub(crate) trait ToWord {
    fn to_word_signed(&self) -> u32;
    fn to_word_zeroed(&self) -> u32;
}

impl ToWord for u8 {
    #[inline]
    fn to_word_signed(&self) -> u32 {
        *self as i8 as i32 as u32
    }

    #[inline]
    fn to_word_zeroed(&self) -> u32 {
        *self as u32
    }
}

impl ToWord for u16 {
    #[inline]
    fn to_word_signed(&self) -> u32 {
        *self as i16 as i32 as u32
    }

    #[inline]
    fn to_word_zeroed(&self) -> u32 {
        *self as u32
    }
}

impl ToWord for u32 {
    #[inline]
    fn to_word_signed(&self) -> u32 {
        *self as i32 as u32
    }

    #[inline]
    fn to_word_zeroed(&self) -> u32 {
        *self as u32
    }
}

pub(crate) trait Address: TryInto<PhysAddr> + core::fmt::Debug + 'static + Copy {}

impl<T> Address for T where T: TryInto<PhysAddr> + core::fmt::Debug + 'static + Copy {}

impl Memory {
    #[instrument(err, skip(self))]
    pub(crate) fn try_read<T: MemRead>(&self, addr: impl Address) -> Result<T, MemReadError> {
        let addr = addr.try_into().map_err(|_| MemReadError::unmapped(addr))?;
        let addr = addr.as_usize();
        let slice = self
            .0
            .get(addr..(addr + size_of::<T>()))
            .ok_or(MemReadError::OutOfBoundsRead(addr as u32))?;
        let value = T::from_slice(slice).map_err(MemReadError::DerefErr)?;
        Ok(value)
    }
    pub(crate) fn read<T: MemRead>(&self, addr: impl Address) -> T {
        self.try_read(addr).unwrap()
    }
    #[instrument(err, skip(self, value))]
    pub(crate) fn try_write<T: MemWrite>(
        &mut self,
        addr: impl Address,
        value: T,
    ) -> Result<(), MemWriteError>
    where
        [(); size_of::<T>()]:,
    {
        let addr = addr.try_into().map_err(|_| MemWriteError::unmapped(addr))?;
        let addr = addr.as_usize();
        let slice = self
            .0
            .get_mut(addr..(addr + size_of::<T>()))
            .ok_or(MemWriteError::OutOfBoundsWrite(addr as u32))?;
        T::write(slice, &value)
    }
    pub(crate) fn write<T: MemWrite>(&mut self, addr: impl Address, value: T)
    where
        [(); size_of::<T>()]:,
    {
        self.try_write(addr, value).unwrap();
    }
    pub(crate) fn try_write_all<I, A, T>(&mut self, start: A, iter: I) -> Result<(), MemWriteError>
    where
        I: IntoIterator<Item = T>,
        T: MemWrite,
        A: Address + Add<u32, Output = A>,
        [(); size_of::<T>()]:,
    {
        let offset = size_of::<I::Item>() as u32;
        for (i, value) in iter.into_iter().enumerate() {
            let i = i as u32;
            self.try_write(start + i * offset, value)?;
        }
        Ok(())
    }
    pub(crate) fn write_all<I, A, T>(&mut self, start: A, iter: I)
    where
        I: IntoIterator<Item = T>,
        T: MemWrite,
        A: Address + Add<u32, Output = A>,
        [(); size_of::<T>()]:,
    {
        self.try_write_all(start, iter).unwrap();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PhysAddr(u32);

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
    #[instrument(err)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KSEG0Addr(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KSEG1Addr(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addr(pub u32);

impl KSEG0Addr {
    pub const fn to_phys(self) -> PhysAddr {
        PhysAddr(self.0 - 0x8000_0000u32)
    }
}

impl KSEG1Addr {
    pub const fn to_phys(self) -> PhysAddr {
        PhysAddr(self.0 - 0xA000_0000u32)
    }
}

impl From<KSEG0Addr> for PhysAddr {
    fn from(value: KSEG0Addr) -> Self {
        value.to_phys()
    }
}

impl From<KSEG1Addr> for PhysAddr {
    fn from(value: KSEG1Addr) -> Self {
        value.to_phys()
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

#[cfg(test)]
mod memory_tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use pretty_assertions::assert_matches;
    #[allow(unused_imports)]
    use pretty_assertions::assert_ne;

    #[test]
    fn read_write_u8_kseg0() {
        let mut mem = Memory::default();
        let addr = 0x8000_1234.map();
        mem.write(addr, 0xABu8);
        assert_eq!(mem.read::<u8>(addr), 0xAB);
    }

    #[test]
    fn read_write_u16_u32_kseg1() {
        let mut mem = Memory::default();
        let addr0 = 0x8000_1000.map();
        let addr1 = 0xA000_1000.map();

        mem.write(addr0, 0x1234u16);
        assert_eq!(mem.read::<u16>(addr1), 0x1234);

        mem.write(addr1, 0xDEADBEEFu32);
        assert_eq!(mem.read::<u32>(addr0), 0xDEADBEEF);
    }

    #[test]
    fn unmapped_address_returns_error() {
        let mut mem = Memory::default();
        use super::{MemReadError, MemWriteError};

        assert_matches!(
            mem.try_read::<u8>(Addr(0x0000_0000)),
            Err(MemReadError::UnmappedRead(_))
        );
        assert_matches!(
            mem.try_write(Addr(0x0000_0000), 0x12u8),
            Err(MemWriteError::UnmappedWrite(_))
        );
    }

    #[test]
    fn out_of_bounds_read_write() {
        let mut mem = Memory::default();
        let phys_size = mem.0.len();
        let base_kseg0 = 0x8000_0000; // start of KSEG0

        // Pick an address near the end of RAM to trigger OutOfBounds
        let addr = base_kseg0 + (phys_size as u32) - 1;
        let addr = addr.map();

        assert!(matches!(
            mem.try_read::<u32>(addr),
            Err(super::MemReadError::OutOfBoundsRead(_))
        ));
        assert!(matches!(
            mem.try_write(addr, 0x1234u16),
            Err(super::MemWriteError::OutOfBoundsWrite(_))
        ));
    }
}
#[cfg(test)]
mod sign_extension_tests {
    use super::*;

    #[test]
    fn test_u8_to_word() {
        let a: u8 = 0x7F; // 127
        let b: u8 = 0xFF; // 255 -> -1 as i8

        // Signed extension
        assert_eq!(a.to_word_signed(), 0x0000007F);
        assert_eq!(b.to_word_signed(), 0xFFFFFFFF);

        // Zero extension
        assert_eq!(a.to_word_zeroed(), 0x0000007F);
        assert_eq!(b.to_word_zeroed(), 0x000000FF);
    }

    #[test]
    fn test_u16_to_word() {
        let a: u16 = 0x7FFF; // 32767
        let b: u16 = 0xFFFF; // 65535 -> -1 as i16

        // Signed extension
        assert_eq!(a.to_word_signed(), 0x00007FFF);
        assert_eq!(b.to_word_signed(), 0xFFFFFFFF);

        // Zero extension
        assert_eq!(a.to_word_zeroed(), 0x00007FFF);
        assert_eq!(b.to_word_zeroed(), 0x0000FFFF);
    }

    #[test]
    fn test_u32_to_word() {
        let a: u32 = 0x12345678;
        let b: u32 = 0xFFFFFFFF;

        // Signed extension (no-op)
        assert_eq!(a.to_word_signed(), 0x12345678);
        assert_eq!(b.to_word_signed(), 0xFFFFFFFF);

        // Zero extension (no-op)
        assert_eq!(a.to_word_zeroed(), 0x12345678);
        assert_eq!(b.to_word_zeroed(), 0xFFFFFFFF);
    }
}
