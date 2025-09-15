use std::ops::{Add, Index, Mul};

use thiserror::Error;
use tracing::instrument;

use crate::cpu::ops;

pub const fn kb(value: usize) -> usize {
    value * 1024
}

pub const fn from_kb(value: usize) -> usize {
    value / 1024
}

pub fn buffer(size: usize) -> Box<[u8]> {
    vec![0u8; size].into_boxed_slice()
}

pub type Buffer = Box<[u8]>;

#[derive(derive_more::Debug)]
#[debug("memory:{}kb", MEM_SIZE/1024)]
pub struct Memory(Buffer);

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

static MEM_SIZE: usize = kb(2048) + kb(8192) + kb(1) + kb(8) + kb(8) + kb(2048) + kb(512) + 512;
// const MEM_SIZE: usize = 600 * 1024 * 1024;
static MEM_KB: usize = from_kb(MEM_SIZE) + 1;

impl Default for Memory {
    fn default() -> Self {
        Memory(buffer(MEM_SIZE))
    }
}

#[derive(derive_more::Debug)]
#[derive_const(Default)]
#[debug("0x{:08X}:0x{:08X}", self.host_start, self.phys_start)]
pub struct MemoryRegion {
    pub host_start: u32,
    pub phys_start: u32,
}

pub const fn memory_map() -> [MemoryRegion; 1 << 16] {
    const REGIONS: [(u32, usize); 8] = [
        (0, kb(2048)),
        (0x1F000000, kb(8192)),
        (0x1F800000, kb(1)),
        (0x1F801000, kb(8)),
        (0x1F802000, kb(8)),
        (0x1FA00000, kb(2048)),
        (0x1FC00000, kb(512)),
        (0x1FFE0000, 512),
    ];
    let mut start = 0u32;
    let mut current = 0;
    let mut table = [const { MemoryRegion::default() }; 1 << 16];
    while current < REGIONS.len() {
        let mut i = REGIONS[current].0 >> 16;
        let end = (REGIONS[current].0 + REGIONS[current].1 as u32) >> 16;
        while i <= end {
            table[i as usize] = MemoryRegion {
                host_start: start,
                phys_start: REGIONS[current].0,
            };
            i += 1;
        }
        start += REGIONS[current].1 as u32;
        current += 1;
    }
    // while current < table.len() {
    //     table[current] = MemoryRegion {
    //         host_start: current as u32,
    //         phys_start: current as u32,
    //     };
    //     current += 1;
    // }
    table
}

pub static MEM_MAP: [MemoryRegion; 1 << 16] = memory_map();

#[inline]
pub const fn map_physical(phys: PhysAddr) -> u32 {
    phys.0 >> 16
}

pub fn lookup_phys(phys: PhysAddr) -> u32 {
    let index = map_physical(phys);
    let region = &MEM_MAP[index as usize];
    let offset = phys.0 - region.phys_start;
    region.host_start + offset
}

impl Index<PhysAddr> for Memory {
    type Output = u8;

    fn index(&self, index: PhysAddr) -> &Self::Output {
        &self.0[lookup_phys(index) as usize]
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
        PhysAddr(self.0 & 0x1FFF_FFFF)
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
        PhysAddr(self.0 & 0x1FFF_FFFF)
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

    // #[test]
    // #[should_panic]
    // fn unmapped_address_panics() {
    //     PhysAddr::map(0x0000_0000); // KUSEG (unmapped)
    // }

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
    #[error("out of bounds write at address 0x{0:08X}")]
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

pub trait MemRead: Sized {
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
        Ok(u16::from_le_bytes(*buf))
    }
}

impl MemRead for u32 {
    fn from_slice(buf: &[u8]) -> Result<u32, DerefError> {
        let buf = buf.as_array().ok_or(DerefError)?;
        Ok(u32::from_le_bytes(*buf))
    }
}

impl MemRead for ops::OpCode {
    fn from_slice(buf: &[u8]) -> Result<Self, DerefError> {
        let buf = buf.as_array().ok_or(DerefError)?;
        Ok(ops::OpCode(u32::from_le_bytes(*buf)))
    }
}

pub trait MemWrite<const N: usize = { size_of::<Self>() }>: Sized {
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
        self.to_le_bytes()
    }
}

impl MemWrite for u32 {
    #[inline]
    fn to_bytes(&self) -> [u8; 4] {
        self.to_le_bytes()
    }
}

impl MemWrite for ops::OpCode {
    #[inline]
    fn to_bytes(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

pub trait ToWord {
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
        *self
    }
}

#[rustfmt::skip]
pub const trait Address: TryInto<PhysAddr> + core::fmt::Debug + 'static + Copy {}

impl<T> const Address for T where T: TryInto<PhysAddr> + core::fmt::Debug + 'static + Copy {}

impl Memory {
    #[instrument(err, skip(self))]
    pub fn try_read<T: MemRead>(&self, addr: impl Address) -> Result<T, MemReadError> {
        let addr = addr.try_into().map_err(|_| MemReadError::unmapped(addr))?;
        let addr = lookup_phys(addr) as usize;
        let slice = self
            .0
            .get(addr..(addr + size_of::<T>()))
            .ok_or(MemReadError::OutOfBoundsRead(addr as u32))?;
        let value = T::from_slice(slice).map_err(MemReadError::DerefErr)?;
        Ok(value)
    }
    pub fn read<T: MemRead>(&self, addr: impl Address) -> T {
        self.try_read(addr).unwrap()
    }
    #[instrument(err, skip(self, value))]
    pub fn try_write<T: MemWrite>(
        &mut self,
        addr: impl Address,
        value: T,
    ) -> Result<(), MemWriteError>
    where
        [(); size_of::<T>()]:,
    {
        let addr: PhysAddr = addr.try_into().map_err(|_| MemWriteError::unmapped(addr))?;
        let addr = lookup_phys(addr) as usize;
        let slice = self
            .0
            .get_mut(addr..(addr + size_of::<T>()))
            .ok_or(MemWriteError::OutOfBoundsWrite(addr as u32))?;
        T::write(slice, &value)
    }
    pub fn write<T: MemWrite>(&mut self, addr: impl Address, value: T)
    where
        [(); size_of::<T>()]:,
    {
        self.try_write(addr, value).unwrap();
    }
    pub fn try_write_all<I, A, T>(&mut self, start: A, iter: I) -> Result<(), MemWriteError>
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
    pub fn write_all<I, A, T>(&mut self, start: A, iter: I)
    where
        I: IntoIterator<Item = T>,
        T: MemWrite,
        A: Address + Add<u32, Output = A>,
        [(); size_of::<T>()]:,
    {
        self.try_write_all(start, iter).unwrap();
    }

    pub fn try_write_array<T: MemWrite<N>, const N: usize>(
        &mut self,
        start: impl Address,
        value: &[T],
    ) -> Result<(), MemWriteError> {
        let start = start
            .try_into()
            .map_err(|_| MemWriteError::unmapped(start))?;
        let start = start.as_usize();
        for (idx, v) in value.iter().enumerate() {
            let start = start + idx * N;
            let end = start + N;
            T::write(&mut self.as_mut()[start..end], v)?;
        }
        Ok(())
    }
    pub fn write_array<T: MemWrite<N>, const N: usize>(
        &mut self,
        start: impl Address,
        value: &[T],
    ) {
        self.try_write_array(start, value).unwrap()
    }
}
