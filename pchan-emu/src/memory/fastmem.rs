use crate::{
    Bus, Emu,
    io::UnhandledIO,
    memory::{MEM_MAP, kb},
};

const PAGE_COUNT: usize = 0x10000;
const PAGE_SIZE: usize = kb(64);

type PageTable = [Option<u32>; PAGE_COUNT];

pub struct Lut {
    pub read:  PageTable,
    pub write: PageTable,
}

pub static LUT: Lut = generate_page_tables();

const fn generate_page_tables() -> Lut {
    let mut table_read: PageTable = [None; PAGE_COUNT];
    let mut table_write: PageTable = [None; PAGE_COUNT];

    const RAM_PAGE_COUNT: usize = kb(2048) / PAGE_SIZE;

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

    const BIOS_PAGE_COUNT: usize = kb(512) / PAGE_SIZE;

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
pub fn util_fast_map_address(address: u32) -> Option<u32> {
    let page = address >> 16;
    let offset = address & 0xFFFF;
    let lut_ptr = LUT.read[page as usize];
    lut_ptr.map(|region_ptr| region_ptr + offset)
}

pub type FastmemResult<T> = Result<T, UnhandledIO>;

pub trait Fastmem: Bus {
    fn read<T: Copy>(&self, address: u32) -> FastmemResult<T>;
    fn write<T: Copy>(&mut self, address: u32, value: T) -> FastmemResult<()>;
}

impl Fastmem for Emu {
    fn read<T: Copy>(&self, address: u32) -> FastmemResult<T> {
        let page = address >> 16;
        let offset = address & 0x0000_FFFF;

        let lut_ptr = LUT.read[page as usize];
        let mem = self.mem().buf.as_ptr();

        match lut_ptr {
            // fastmem
            Some(region_ptr) => unsafe {
                let ptr = mem.add(region_ptr as usize).add(offset as usize);
                Ok(std::ptr::read(ptr as *const T))
            },
            // memcheck
            None => Err(UnhandledIO(address)),
        }
    }

    fn write<T: Copy>(&mut self, address: u32, value: T) -> FastmemResult<()> {
        if self.cpu().isc() {
            return Ok(());
        }

        let page = address >> 16;
        let offset = address & 0x0000_FFFF;

        let lut_ptr = LUT.write[page as usize];
        let mem = self.mem_mut().buf.as_mut_ptr();

        if let Some(region_ptr) = lut_ptr {
            self.dynarec_cache.invalidate(address);
            unsafe {
                let ptr = mem.add(region_ptr as usize).add(offset as usize);
                std::ptr::write(ptr as *mut _, value);
            }
            Ok(())
        } else {
            Err(UnhandledIO(address))
        }
    }
}
