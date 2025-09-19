use std::sync::LazyLock;

use const_hex::const_encode;
use pchan_utils::hex;
use tracing::instrument;

use crate::memory::{Memory, kb};

const PAGE_COUNT: usize = 0x10000;
const PAGE_SIZE: usize = kb(64);

type PageTable = Box<[Option<u32>; PAGE_COUNT]>;

pub struct Lut {
    read: PageTable,
    write: PageTable,
}

static LUT: LazyLock<Lut> = LazyLock::new(generate_page_tables);

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

    const BIOS_PAGE_COUNT: usize = kb(512) / PAGE_SIZE;
    const BIOS_OFFSET: u32 = (kb(2048) + kb(8192) + kb(64) + kb(64) + kb(64) + kb(2048)) as u32;

    for i in 0..BIOS_PAGE_COUNT {
        let offset = ((i * PAGE_SIZE) & 0x1FFFFF) as u32;

        #[allow(clippy::identity_op)]
        table_read[i + 0x1FC0] = Some(BIOS_OFFSET + offset);
        table_read[i + 0x9FC0] = Some(BIOS_OFFSET + offset);
        table_read[i + 0xBFC0] = Some(BIOS_OFFSET + offset);

        // thats it, bios is not writeable
    }

    // DONE: make sure to increase emulator memory to use a 64kb scratch
    // this is so we can map scratch with fastmem
    const SCRATCH_OFFSET: u32 = (kb(2048) + kb(8129)) as u32;

    #[allow(clippy::identity_op)]
    table_read[0x1F80] = Some(SCRATCH_OFFSET);
    table_read[0x9F80] = Some(SCRATCH_OFFSET);
    // scratch cannot be accesed from kseg1
    #[allow(clippy::identity_op)]
    table_write[0x1F80] = Some(SCRATCH_OFFSET);
    table_write[0x9F80] = Some(SCRATCH_OFFSET);

    Lut {
        read: table_read,
        write: table_write,
    }
}

impl Memory {
    /// # Safety
    ///
    /// this is never safe, live fast die young
    pub unsafe fn read_raw<T: Copy>(mem: *const u8, address: u32) -> T {
        let page = address >> 16;

        let lut_ptr = LUT.read[page as usize];

        match lut_ptr {
            // fastmem
            Some(region_ptr) => {
                let offset = address & 0xFFFF;
                let ptr = mem
                    .wrapping_add(region_ptr as usize)
                    .wrapping_add(offset as usize);

                unsafe { *(ptr as *const T) }
            }
            // memcheck
            None => {
                tracing::error!(
                    "read to address=0x{address:08X?} requires memcheck (not yet implemented)"
                );
                unsafe { std::mem::zeroed() }
            }
        }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[instrument(skip(address), fields(address = %hex(&address)))]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn read32(mem: *const u8, address: u32) -> i32 {
        unsafe { Memory::read_raw::<i32>(mem, address) }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address)))]
    pub unsafe extern "C" fn readi16(mem: *const u8, address: u32) -> i32 {
        unsafe { Memory::read_raw::<i16>(mem, address) as i32 }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address)))]
    pub unsafe extern "C" fn readi8(mem: *const u8, address: u32) -> i32 {
        unsafe { Memory::read_raw::<i8>(mem, address) as i32 }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address)))]
    pub unsafe extern "C" fn readu16(mem: *const u8, address: u32) -> i32 {
        unsafe { Memory::read_raw::<u16>(mem, address) as u32 as i32 }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address)))]
    pub unsafe extern "C" fn readu8(mem: *const u8, address: u32) -> i32 {
        unsafe { Memory::read_raw::<u8>(mem, address) as u32 as i32 }
    }
}

impl Memory {
    /// # write
    ///
    /// aligned write generic over T.
    /// checks for alignment against T itself, so simd instructions are valid.
    ///
    /// # SAFETY
    ///
    /// here be segfaults... should be fine tho
    pub unsafe fn write_raw<T>(mem: *mut u8, address: u32, value: T) {
        let page = address >> 16;

        let lut_ptr = LUT.read[page as usize];

        match lut_ptr {
            // fastmem
            Some(region_ptr) => {
                let offset = address & 0xFFFF;
                let ptr = mem
                    .wrapping_add(region_ptr as usize)
                    .wrapping_add(offset as usize);
                let ptr = ptr as *mut T;

                if !ptr.is_aligned() {
                    tracing::error!(
                        "emulator trigger unaligned write in an aligned operation. consider the exception handler invoked (not yet implemented)"
                    );
                    return;
                }

                unsafe {
                    std::ptr::write(ptr, value);
                }
            }
            // memcheck
            None => {
                tracing::error!(
                    "write to address={address:08X?} requires memcheck (not yet implemented)"
                );
            }
        }
    }
    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address), value = %hex(&value)))]
    pub unsafe extern "C" fn write32(mem: *mut u8, address: u32, value: i32) {
        unsafe { Memory::write_raw(mem, address, value) }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address), value = %hex(&value)))]
    pub unsafe extern "C" fn write16(mem: *mut u8, address: u32, value: i32) {
        unsafe { Memory::write_raw(mem, address, value as i16) }
    }

    /// # Safety
    ///
    /// this is never safe, live fast die young
    #[unsafe(no_mangle)]
    #[instrument(skip(address), fields(address = %hex(&address), value = %hex(&value)))]
    pub unsafe extern "C" fn write8(mem: *mut u8, address: u32, value: i32) {
        unsafe { Memory::write_raw(mem, address, value as i8) }
    }
}
