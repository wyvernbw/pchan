use std::{
    marker::PhantomData,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

use thiserror::Error;

use crate::{
    MAX_SIMD_WIDTH,
    cpu::{Cpu, ops},
};

pub mod fastmem;

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
pub struct Memory {
    pub buf: Buffer,
    pub cache: Buffer,
}

pub struct MemMap {
    ram: usize,
    scratch: usize,
    io: usize,
    exp_2: usize,
    exp_3: usize,
    bios: usize,
    cache_control: usize,
}

pub static MEM_MAP: MemMap = MemMap {
    ram: 0,
    scratch: kb(2048),
    io: kb(2048) + kb(64),
    exp_2: kb(2048) + kb(64) + kb(64),
    exp_3: kb(2048) + kb(64) + kb(64) + kb(64),
    bios: kb(2048) + kb(64) + kb(64) + kb(64) + kb(2048),
    cache_control: kb(2048) + kb(64) + kb(64) + kb(64) + kb(2048) + kb(512),
};

static MEM_SIZE: usize = kb(2048) + kb(64) + kb(64) + kb(64) + kb(2048) + kb(512) + kb(64);
// const MEM_SIZE: usize = 600 * 1024 * 1024;
static MEM_KB: usize = from_kb(MEM_SIZE) + 1;

impl Default for Memory {
    fn default() -> Self {
        Memory {
            buf: buffer(MEM_SIZE),
            cache: buffer(4096),
        }
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

pub struct Sign;
pub struct Zero;
pub struct NoExt;

pub const trait Extend<E> {
    type Out;
    fn ext(self) -> Self::Out;
}

impl<T> const Extend<NoExt> for T {
    type Out = Self;

    fn ext(self) -> Self::Out {
        self
    }
}

impl const Extend<Sign> for u8 {
    type Out = i32;
    fn ext(self) -> Self::Out {
        self as i8 as i32
    }
}

impl const Extend<Zero> for u8 {
    type Out = u32;
    fn ext(self) -> Self::Out {
        self as u32
    }
}

impl const Extend<Sign> for u16 {
    type Out = i32;
    fn ext(self) -> Self::Out {
        self as i16 as i32
    }
}

impl const Extend<Zero> for u16 {
    type Out = u32;
    fn ext(self) -> Self::Out {
        self as u32
    }
}

impl const Extend<Sign> for i8 {
    type Out = i32;
    fn ext(self) -> Self::Out {
        self as i32
    }
}

impl const Extend<Zero> for i8 {
    type Out = u32;
    fn ext(self) -> Self::Out {
        self as u8 as u32
    }
}

impl const Extend<Sign> for i16 {
    type Out = i32;
    fn ext(self) -> Self::Out {
        self as i32
    }
}

impl const Extend<Zero> for i16 {
    type Out = u32;
    fn ext(self) -> Self::Out {
        self as u16 as u32
    }
}

pub struct Chunk<El>(PhantomData<El>);
impl<El> Chunk<El> {
    pub const LANE_COUNT: usize = MAX_SIMD_WIDTH / size_of::<El>();
}

impl Memory {
    pub fn read<T, E>(&self, cpu: &Cpu, address: u32) -> T::Out
    where
        T: Extend<E> + Copy,
    {
        let read = unsafe { self.read_raw(cpu, address) };
        T::ext(read)
    }

    pub fn write<T: Copy>(&mut self, cpu: &Cpu, address: u32, value: T) {
        unsafe {
            self.write_raw(cpu, address, value);
        }
    }

    pub fn write_many<T: SimdElement>(&mut self, cpu: &Cpu, address: u32, values: &[T])
    where
        LaneCount<{ Chunk::<T>::LANE_COUNT }>: SupportedLaneCount,
    {
        let data = values.chunks_exact(Chunk::<T>::LANE_COUNT);
        let remaining = data.remainder();
        let data = data.map(|data| Simd::<T, { Chunk::<T>::LANE_COUNT }>::from_slice(data));

        let mut ptr = 0;
        for el in data {
            self.write(cpu, address + ptr, el);

            let offset = size_of_val(&el) as u32;
            ptr += offset;
        }

        for el in remaining.iter().cloned() {
            self.write(cpu, address + ptr, el);

            let offset = size_of_val(&el) as u32;
            ptr += offset;
        }
    }
}

pub mod ext {
    pub use super::NoExt;
    pub use super::Sign;
    pub use super::Zero;

    use super::Extend;

    pub const fn extend<T, E>(value: T) -> T::Out
    where
        T: const Extend<E>,
    {
        value.ext()
    }

    pub const fn sign<T>(value: T) -> T::Out
    where
        T: const Extend<Sign>,
    {
        value.ext()
    }

    pub const fn zero<T>(value: T) -> T::Out
    where
        T: const Extend<Zero>,
    {
        value.ext()
    }
}
