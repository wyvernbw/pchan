pub mod fastmem;

pub const fn kb(value: usize) -> usize {
    value * 1024
}

pub const fn mb(value: usize) -> usize {
    kb(value) * 1024
}

pub const fn from_kb(value: usize) -> usize {
    value / 1024
}

pub fn buffer(size: usize) -> Box<[u8]> {
    vec![0x0; size].into_boxed_slice()
}

pub type Buffer = Box<[u8]>;

#[derive(derive_more::Debug, Clone)]
#[debug("memory:{}kb", MEM_SIZE/1024)]
pub struct MemoryState {
    pub buf: Buffer,
}

pub struct MemMap {
    pub ram: usize,
    pub scratch: usize,
    pub io: usize,
    pub exp_2: usize,
    pub exp_3: usize,
    pub bios: usize,
    pub cache_control: usize,
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

pub static GUEST_MEM_MAP: MemMap = MemMap {
    ram: 0,
    scratch: 0x1F80_0000,
    io: 0x1f801000,
    exp_2: 0x1f802000,
    exp_3: 0x1fa00000,
    bios: 0x1fc00000,
    cache_control: 0xfffe0000,
};

static MEM_SIZE: usize = kb(2048) + kb(64) + kb(64) + kb(64) + kb(2048) + kb(512) + kb(64);
// const MEM_SIZE: usize = 600 * 1024 * 1024;
static MEM_KB: usize = from_kb(MEM_SIZE) + 1;

impl Default for MemoryState {
    fn default() -> Self {
        MemoryState {
            buf: buffer(MEM_SIZE),
        }
    }
}

impl MemoryState {
    #[inline(always)]
    pub fn read_region<T: Copy>(&self, host_region: usize, guest_region: usize, address: u32) -> T {
        let offset = (address & 0x1fff_ffff) as usize - (guest_region & 0x1fff_ffff);
        let idx = host_region + offset;
        let slice = &self.buf[idx..];
        unsafe { slice.as_ptr().cast::<T>().read() }
    }
    #[inline(always)]
    pub fn write_region<T: Copy>(
        &mut self,
        host_region: usize,
        guest_region: usize,
        address: u32,
        value: T,
    ) {
        let offset = (address & 0x1fff_ffff) as usize - (guest_region & 0x1fff_ffff);
        let idx = host_region + offset;
        let slice = &mut self.buf[idx..];
        unsafe {
            slice.as_mut_ptr().cast::<T>().write(value);
        }
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

    #[inline(always)]
    fn ext(self) -> Self::Out {
        self
    }
}

impl const Extend<Sign> for u8 {
    type Out = i32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as i8 as i32
    }
}

impl const Extend<Zero> for u8 {
    type Out = u32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as u32
    }
}

impl const Extend<Sign> for u16 {
    type Out = i32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as i16 as i32
    }
}

impl const Extend<Zero> for u16 {
    type Out = u32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as u32
    }
}

impl const Extend<Sign> for i8 {
    type Out = i32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as i32
    }
}

impl const Extend<Zero> for i8 {
    type Out = u32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as u8 as u32
    }
}

impl const Extend<Sign> for i16 {
    type Out = i32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as i32
    }
}

impl const Extend<Zero> for i16 {
    type Out = u32;
    #[inline(always)]
    fn ext(self) -> Self::Out {
        self as u16 as u32
    }
}

pub mod ext {
    pub use super::NoExt;
    pub use super::Sign;
    pub use super::Zero;

    use super::Extend;

    #[inline(always)]
    pub const fn extend<T, E>(value: T) -> T::Out
    where
        T: const Extend<E>,
    {
        value.ext()
    }

    #[inline(always)]
    pub const fn sign<T>(value: T) -> T::Out
    where
        T: const Extend<Sign>,
    {
        value.ext()
    }

    #[inline(always)]
    pub const fn zero<T>(value: T) -> T::Out
    where
        T: const Extend<Zero>,
    {
        value.ext()
    }
}
