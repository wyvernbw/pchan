use core::alloc::Allocator;
use core::ops::Range;
use core::{ptr, slice};

use crate::Emu;
use crate::io::UnhandledIO;
use crate::memory::{MEM_MAP, kb};

const PAGE_COUNT: usize = 0x10000;
const PAGE_SIZE: usize = kb(64);

type PageTable = [Option<u32>; PAGE_COUNT];

pub struct Lut {
    pub read:  PageTable,
    pub write: PageTable,
}

pub static LUT: Lut = generate_page_tables();

#[allow(clippy::large_stack_arrays)]
const fn generate_page_tables() -> Lut {
    const RAM_PAGE_COUNT: usize = kb(2048) / PAGE_SIZE;
    const BIOS_PAGE_COUNT: usize = kb(512) / PAGE_SIZE;

    let mut table_read: PageTable = [None; PAGE_COUNT];
    let mut table_write: PageTable = [None; PAGE_COUNT];

    // for each region, it will map `kuseg`, `kseg0` and `kseg1` respectively

    let mut i = 0;
    while i < RAM_PAGE_COUNT {
        let offset = ((i * PAGE_SIZE) & 0x1FFFFF) as u32;

        #[allow(clippy::identity_op)]
        table_read[i + 0x0000] = Some(offset);
        table_read[i + 0x8000] = Some(offset);
        table_read[i + 0xA000] = Some(offset);

        #[allow(clippy::identity_op)]
        table_write[i + 0x0000] = Some(offset);
        table_write[i + 0x8000] = Some(offset);
        table_write[i + 0xA000] = Some(offset);

        i += 1;
    }

    // map 8MB expansion region 1 to ram by default

    i = 0;
    while i < RAM_PAGE_COUNT * 4 {
        let offset = (i * PAGE_SIZE) as u32;
        table_read[i + 0x1F00] = Some(offset);
        table_read[i + 0x9F00] = Some(offset);
        table_read[i + 0xBF00] = Some(offset);

        table_write[i + 0x1F00] = Some(offset);
        table_write[i + 0x9F00] = Some(offset);
        table_write[i + 0xBF00] = Some(offset);

        i += 1;
    }

    i = 0;
    while i < BIOS_PAGE_COUNT {
        let offset = (i * PAGE_SIZE) as u32;

        table_read[i + 0x1FC0] = Some(MEM_MAP.bios as u32 + offset);
        table_read[i + 0x9FC0] = Some(MEM_MAP.bios as u32 + offset);
        table_read[i + 0xBFC0] = Some(MEM_MAP.bios as u32 + offset);

        // thats it, bios is not writeable

        i += 1;
    }

    Lut {
        read:  table_read,
        write: table_write,
    }
}

#[inline(always)]
#[must_use]
pub fn util_fast_map_address(address: u32) -> Option<u32> {
    let page = address >> 16;
    let offset = address & 0xFFFF;
    let lut_ptr = LUT.read[page as usize];
    lut_ptr.map(|region_ptr| region_ptr + offset)
}

pub type FastmemResult<T> = Result<T, UnhandledIO>;

impl<A: Allocator> Emu<A> {
    fn fastmem_ptr<T: Copy>(&self, address: u32) -> FastmemResult<*const T> {
        let page = address >> 16;
        let offset = address & 0x0000_FFFF;

        let lut_ptr = LUT.read[page as usize];
        let mem = self.mem.buf.as_ptr();

        match lut_ptr {
            // fastmem
            Some(region_ptr) => unsafe {
                let ptr = mem.add(region_ptr as usize).add(offset as usize);
                Ok(ptr.cast::<T>())
            },
            // memcheck
            None => Err(UnhandledIO(address)),
        }
    }
    fn fastmem_mut_ptr<T: Copy>(&mut self, address: u32) -> FastmemResult<*mut T> {
        let page = address >> 16;
        let offset = address & 0x0000_FFFF;

        let lut_ptr = LUT.write[page as usize];
        let mem = self.mem.buf.as_mut_ptr();

        match lut_ptr {
            // fastmem
            Some(region_ptr) => unsafe {
                let ptr = mem.add(region_ptr as usize).add(offset as usize);
                Ok(ptr.cast::<T>())
            },
            // memcheck
            None => Err(UnhandledIO(address)),
        }
    }

    pub fn fastmem_read<T: Copy>(&self, address: u32) -> FastmemResult<T> {
        let ptr = self.fastmem_ptr::<T>(address)?;
        unsafe { Ok(ptr::read(ptr)) }
    }

    pub fn fastmem_write<T: Copy>(&mut self, address: u32, value: T) -> FastmemResult<()> {
        if self.cpu.isc() {
            return Ok(());
        }
        let ptr = self.fastmem_mut_ptr(address)?;
        self.dynarec_cache.invalidate(address);
        unsafe {
            ptr::write(ptr, value);
        }
        Ok(())
    }

    pub fn fastmem_slice<T: Copy>(&self, range: impl Into<Range<u32>>) -> FastmemResult<&[T]> {
        let range = range.into();
        let start = self.fastmem_ptr::<T>(range.start)?;
        let _ = self.fastmem_ptr::<T>(range.end - 1)?;
        let bytes = range.len();
        let len = bytes / size_of::<T>();
        debug_assert!(start.is_aligned());
        let slice = unsafe { slice::from_raw_parts(start, len) };
        Ok(slice)
    }

    pub fn fastmem_slice_mut<T: Copy>(
        &mut self,
        range: impl Into<Range<u32>>,
    ) -> FastmemResult<&mut [T]> {
        let range = range.into();
        let start = self.fastmem_mut_ptr::<T>(range.start)?;
        let _ = self.fastmem_mut_ptr::<T>(range.end - 1)?;
        let bytes = range.len();
        let len = bytes / size_of::<T>();
        debug_assert!(start.is_aligned());

        for addr in range.clone().step_by(4) {
            self.dynarec_cache.invalidate(addr);
        }

        let slice = unsafe { slice::from_raw_parts_mut(start, len) };
        Ok(slice)
    }
}
