use std::sync::LazyLock;

use pchan_utils::hex;
use tracing::{Level, instrument};

use crate::{
    cpu::Cpu,
    memory::{MEM_MAP, Memory, kb},
};

const PAGE_COUNT: usize = 0x10000;
const PAGE_SIZE: usize = kb(64);

type PageTable = Box<[Option<u32>; PAGE_COUNT]>;

pub struct Lut {
    pub read: PageTable,
    pub write: PageTable,
}

pub static LUT: LazyLock<Lut> = LazyLock::new(generate_page_tables);

fn generate_page_tables() -> Lut {
    let mut table_read: PageTable = Box::new([None; PAGE_COUNT]);
    let mut table_write: PageTable = Box::new([None; PAGE_COUNT]);

    const RAM_PAGE_COUNT: usize = kb(2048) / PAGE_SIZE;

    // for each region, it will map `kuseg`, `kseg0` and `kseg1` respectively

    for i in 0..(RAM_PAGE_COUNT * 4) {
        // RAM is mirrored 4 times
        let offset = ((i * PAGE_SIZE) & 0x1FFFFF) as u32;

        #[allow(clippy::identity_op)]
        table_read[i + 0x0000] = Some(offset);
        table_read[i + 0x8000] = Some(offset);
        table_read[i + 0xA000] = Some(offset);

        #[allow(clippy::identity_op)]
        table_write[i + 0x0000] = Some(offset);
        table_write[i + 0x8000] = Some(offset);
        table_write[i + 0xA000] = Some(offset);
    }

    // map 8MB expansion region 1 to ram by default

    for i in 0..(RAM_PAGE_COUNT * 4) {
        let offset = ((i * PAGE_SIZE) & 0x1FFFFF) as u32;
        #[allow(clippy::identity_op)]
        table_read[i + 0x1F00] = Some(offset);
        table_read[i + 0x9F00] = Some(offset);
        table_read[i + 0xBF00] = Some(offset);

        #[allow(clippy::identity_op)]
        table_write[i + 0x1F00] = Some(offset);
        table_write[i + 0x9F00] = Some(offset);
        table_write[i + 0xBF00] = Some(offset);
    }

    const BIOS_PAGE_COUNT: usize = kb(512) / PAGE_SIZE;

    for i in 0..BIOS_PAGE_COUNT {
        let offset = ((i * PAGE_SIZE) & 0x1FFFFF) as u32;

        #[allow(clippy::identity_op)]
        table_read[i + 0x1FC0] = Some(MEM_MAP.bios as u32 + offset);
        table_read[i + 0x9FC0] = Some(MEM_MAP.bios as u32 + offset);
        table_read[i + 0xBFC0] = Some(MEM_MAP.bios as u32 + offset);

        // thats it, bios is not writeable
    }

    Lut {
        read: table_read,
        write: table_write,
    }
}

impl Memory {
    #[inline(always)]
    pub fn util_fast_map_address(address: u32) -> Option<u32> {
        let page = address >> 16;
        let offset = address & 0xFFFF;
        let lut_ptr = LUT.read[page as usize];
        lut_ptr.map(|region_ptr| region_ptr + offset)
    }
    /// # Safety
    ///
    /// this is never safe, live fast die young
    pub unsafe fn read_raw<T: Copy>(&self, cpu: &Cpu, address: u32) -> T {
        let page = address >> 16;
        let offset = address & 0xFFFF;

        let lut_ptr = LUT.read[page as usize];
        let mem = self.buf.as_ptr();

        match lut_ptr {
            // fastmem
            Some(region_ptr) => {
                if cpu.isc() {
                    let address = address & 0xFFF;
                    unsafe {
                        let ptr = self.cache.as_ptr().add(address as usize);
                        let ptr = ptr as *const T;
                        return std::ptr::read_unaligned(ptr);
                    }
                }

                let ptr = mem
                    .wrapping_add(region_ptr as usize)
                    .wrapping_add(offset as usize);

                unsafe { std::ptr::read_unaligned(ptr as *const T) }
            }
            // memcheck
            None => {
                let _span = tracing::trace_span!("memcheck").entered();
                let address = address & 0x1fff_ffff;
                match address {
                    0xfffe0000.. => {
                        let ptr = mem
                            .wrapping_add(offset as usize)
                            .wrapping_add(MEM_MAP.cache_control);
                        // tracing::trace!("read in cache control");
                        unsafe { *(ptr as *const T) }
                    }
                    0x1f801000..0x1f803000 => {
                        let ptr = mem.wrapping_add(offset as usize).wrapping_add(MEM_MAP.io);
                        match address {
                            0x1f801800..0x1f801804 => {
                                todo!("read from cdrom mapped memory");
                            }
                            0x1f801810..=0x1f801814 => {
                                todo!("read from gpu mapped memory");
                            }
                            0x1f801100..=0x1f801108
                            | 0x1f801110..=0x1f801118
                            | 0x1f801120..=0x1f801128 => {
                                tracing::trace!("read to timer registers");
                                unsafe { *(ptr as *const T) }
                            }
                            _ => unsafe {
                                tracing::error!("read from unsupported io mapped region");
                                *(ptr as *const T)
                            },
                        }
                    }

                    _ => {
                        // panic!("unsupported region for {}!", hex(address));
                        unsafe { std::mem::zeroed() }
                    }
                }
            }
        }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(cpu, self, address), fields(address = %hex(address), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn read32(&self, cpu: &Cpu, address: u32) -> i32 {
        unsafe { self.read_raw::<i32>(cpu, address) }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn readi16(&self, cpu: &Cpu, address: u32) -> i32 {
        unsafe { self.read_raw::<i16>(cpu, address) as i32 }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn readi8(&self, cpu: &Cpu, address: u32) -> i32 {
        unsafe { self.read_raw::<i8>(cpu, address) as i32 }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn readu16(&self, cpu: &Cpu, address: u32) -> i32 {
        unsafe { self.read_raw::<u16>(cpu, address) as u32 as i32 }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn readu8(&self, cpu: &Cpu, address: u32) -> i32 {
        unsafe { self.read_raw::<u8>(cpu, address) as u32 as i32 }
    }
}

impl Memory {
    /// # SAFETY
    pub unsafe fn write_impl<T>(ptr: *mut u8, value: T) {
        let ptr = ptr as *mut T;

        if !ptr.is_aligned() {
            tracing::error!(
                "emulator triggered unaligned write in an aligned operation. consider the exception handler invoked (not yet implemented)"
            );
            unsafe { std::ptr::write_unaligned(ptr, value) };
            return;
        }

        unsafe {
            std::ptr::write(ptr, value);
        }
    }
    /// # write
    ///
    /// aligned write generic over T.
    /// checks for alignment against T itself, so simd instructions are valid.
    ///
    /// # SAFETY
    ///
    /// here be segfaults... should be fine tho
    pub unsafe fn write_raw<T: Copy>(&mut self, cpu: &Cpu, address: u32, value: T) {
        let page = address >> 16;
        let offset = address & 0x0000_FFFF;

        let lut_ptr = LUT.read[page as usize];
        let mem = self.buf.as_mut_ptr();

        match lut_ptr {
            // fastmem
            Some(region_ptr) => {
                // FIXME: cache isolation check on fast path is stupid
                // also not all addresses are cached, so this is straight wrong
                if cpu.isc() {
                    // let address = address & 0xFFF;
                    // unsafe {
                    //     let ptr = self.cache.as_mut_ptr().add(address as usize);
                    //     Memory::write_impl(ptr, value);
                    //     return;
                    // }
                    return;
                }

                let ptr = mem
                    .wrapping_add(region_ptr as usize)
                    .wrapping_add(offset as usize);
                unsafe {
                    Memory::write_impl(ptr, value);
                }
            }
            // memcheck
            None => {
                let _span = tracing::trace_span!("memcheck").entered();
                let address = address & 0x1fff_ffff;
                match address {
                    0x1ffe0000.. => {
                        let ptr = mem
                            .wrapping_add(offset as usize)
                            .wrapping_add(MEM_MAP.cache_control);
                        unsafe {
                            Memory::write_impl(ptr, value);
                        };
                        Memory::write_to_cache_control(address, value);
                    }
                    0x1f801000..0x1f803000 => {
                        let ptr = mem.wrapping_add(offset as usize).wrapping_add(MEM_MAP.io);

                        match address {
                            0x1F801070 => {
                                // I_STAT
                                debug_assert_eq!(size_of::<T>(), 4);
                                unsafe {
                                    Memory::write_impl(ptr, value);
                                }
                            }

                            0x1F801074 => {
                                // I_MASK
                                unsafe {
                                    Memory::write_impl(ptr, value);
                                }
                                // Maybe check if any pending interrupts should now fire
                            }

                            0x1f801800..0x1f801804 => {
                                todo!("write to cdrom mapped memory")
                            }

                            0x1f801810..=0x1f801814 => {
                                todo!("write to gpu mapped memory");
                            }

                            _ => unsafe {
                                Memory::write_impl(ptr, value);
                            },
                        }
                    }
                    _ => {
                        // panic!("unsupported region!");
                    }
                }
            }
        }
    }

    fn write_to_cache_control<T>(address: u32, value: T) {
        match address {
            0xfffe_0130 => {
                tracing::trace!("side effect");
            }
            _ => {
                // tracing::error!("unsupported cache control address");
            }
        }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), value = %hex(value), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn write32(&mut self, cpu: &Cpu, address: u32, value: i32) {
        unsafe { self.write_raw(cpu, address, value) }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), value = %hex(value), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn write16(&mut self, cpu: &Cpu, address: u32, value: i32) {
        unsafe { self.write_raw(cpu, address, value as i16) }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(level = Level::TRACE, skip(self, address, cpu), fields(address = %hex(address), value = %hex(value), isc = cpu.isc()))]
    pub unsafe extern "C-unwind" fn write8(&mut self, cpu: &Cpu, address: u32, value: i32) {
        unsafe { self.write_raw(cpu, address, value as i8) }
    }
}
