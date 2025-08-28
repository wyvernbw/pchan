use thiserror::Error;

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

#[derive(Error, Debug)]
enum MemReadError {
    #[error("read from unmapped address {0}")]
    UnmappedRead(u32),
    #[error(transparent)]
    DerefErr(DerefError),
    #[error("out of bounds read at address {0}")]
    OutOfBoundsRead(u32),
}

#[derive(Error, Debug)]
enum MemWriteError {
    #[error("write to unmapped address {0}")]
    UnmappedWrite(u32),
    #[error(transparent)]
    DerefErr(DerefError),
    #[error("partial write into buffer {0:?} (size: {1}) from buffer {2:?} (size:{3})")]
    MismatchedBuffers(*const u8, usize, *const u8, usize),
    #[error("out of bounds write at address {0}")]
    OutOfBoundsWrite(u32),
}

#[derive(Debug, Error)]
#[error("error dereferencing slice")]
pub struct DerefError;

trait MemRead: Sized {
    fn from_slice(buf: &[u8]) -> Result<Self, DerefError>;
}

impl MemRead for u8 {
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

trait MemWrite: Sized {
    fn to_bytes(&self) -> [u8; size_of::<Self>()];
    fn write(buf: &mut [u8], value: &Self) -> Result<(), MemWriteError>
    where
        [(); size_of::<Self>()]:,
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
    fn to_bytes(&self) -> [u8; 1] {
        [*self]
    }
    fn write(buf: &mut [u8], value: &Self) -> Result<(), MemWriteError> {
        buf[0] = *value;
        Ok(())
    }
}

impl MemWrite for u16 {
    fn to_bytes(&self) -> [u8; 2] {
        self.to_be_bytes()
    }
}

impl MemWrite for u32 {
    fn to_bytes(&self) -> [u8; 4] {
        self.to_be_bytes()
    }
}

impl Memory {
    fn try_read<T: MemRead>(&self, addr: u32) -> Result<T, MemReadError> {
        let addr = PhysAddr::try_new(addr).ok_or(MemReadError::UnmappedRead(addr))?;
        let addr = addr.as_usize();
        let slice = self
            .0
            .get(addr..(addr + size_of::<T>()))
            .ok_or(MemReadError::OutOfBoundsRead(addr as u32))?;
        let value = T::from_slice(slice).map_err(MemReadError::DerefErr)?;
        Ok(value)
    }
    fn read<T: MemRead>(&self, addr: u32) -> T {
        self.try_read(addr).unwrap()
    }
    fn try_write<T: MemWrite>(&mut self, addr: u32, value: T) -> Result<(), MemWriteError>
    where
        [(); size_of::<T>()]:,
    {
        let addr = PhysAddr::try_new(addr).ok_or(MemWriteError::UnmappedWrite(addr))?;
        let addr = addr.as_usize();
        let slice = self
            .0
            .get_mut(addr..(addr + size_of::<T>()))
            .ok_or(MemWriteError::OutOfBoundsWrite(addr as u32))?;
        T::write(slice, &value)
    }
    fn write<T: MemWrite>(&mut self, addr: u32, value: T)
    where
        [(); size_of::<T>()]:,
    {
        self.try_write(addr, value).unwrap();
    }
}

pub struct PhysAddr(u32);

impl PhysAddr {
    fn try_new(address: u32) -> Option<Self> {
        let header = address & 0xE000_0000;
        match header {
            // KSEG0
            0x8000_0000 => Some(PhysAddr(address - 0x8000_0000)),
            // KSEG1
            0xA000_0000 => Some(PhysAddr(address - 0xA000_0000)),
            _ => None,
        }
    }
    fn new(address: u32) -> Self {
        PhysAddr::try_new(address).expect(&format!("unmapped address: {address}"))
    }
    fn as_usize(self) -> usize {
        self.into()
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
        let addr = PhysAddr::new(0x8000_0000);
        assert_eq!(addr.0, 0x0000_0000);

        let addr = PhysAddr::new(0x8020_0100);
        assert_eq!(addr.0, 0x0020_0100);
    }

    #[test]
    fn kseg1_to_physical() {
        let addr = PhysAddr::new(0xA000_0000);
        assert_eq!(addr.0, 0x0000_0000);

        let addr = PhysAddr::new(0xA020_0100);
        assert_eq!(addr.0, 0x0020_0100);
    }

    #[test]
    #[should_panic(expected = "unmapped address")]
    fn unmapped_address_panics() {
        PhysAddr::new(0x0000_0000); // KUSEG (unmapped)
    }

    #[test]
    fn try_new_returns_none_for_unmapped() {
        assert!(PhysAddr::try_new(0x0000_0000).is_none());
        assert!(PhysAddr::try_new(0xC000_0000).is_none()); // KSEG2
    }
}

#[cfg(test)]
mod memory_tests {
    use super::{Memory, PhysAddr};
    use pretty_assertions::assert_eq;
    use pretty_assertions::assert_matches;
    #[allow(unused_imports)]
    use pretty_assertions::assert_ne;

    #[test]
    fn read_write_u8_kseg0() {
        let mut mem = Memory::default();
        let addr = 0x8000_1234;
        mem.write(addr, 0xABu8);
        assert_eq!(mem.read::<u8>(addr), 0xAB);
    }

    #[test]
    fn read_write_u16_u32_kseg1() {
        let mut mem = Memory::default();
        let addr0 = 0x8000_1000;
        let addr1 = 0xA000_1000;

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
            mem.try_read::<u8>(0x0000_0000),
            Err(MemReadError::UnmappedRead(_))
        );
        assert_matches!(
            mem.try_write(0x0000_0000, 0x12u8),
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
