use crate::{Emu, io::IO};

use crate::memory::ext;
impl Emu {
    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn urread32(self: *mut Emu, address: u32, overwrite: u32) -> u32 {
        unsafe { IO::read32_unaligned_r(&mut *self, address, overwrite) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn ulread32(self: *mut Emu, address: u32, overwrite: u32) -> u32 {
        unsafe { IO::read32_unaligned_l(&mut *self, address, overwrite) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn ulwrite32(self: *mut Emu, address: u32, value: u32) {
        unsafe {
            IO::write32_unaligned_l(&mut *self, address, value as _);
        }
    }
    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn urwrite32(self: *mut Emu, address: u32, value: u32) {
        unsafe {
            IO::write32_unaligned_r(&mut *self, address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn write8v2(self: *mut Emu, address: u32, value: i32) {
        unsafe {
            IO::write::<i8>(&mut *self, address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn write16v2(self: *mut Emu, address: u32, value: i32) {
        unsafe {
            IO::write::<i16>(&mut *self, address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn write32v2(self: *mut Emu, address: u32, value: i32) {
        unsafe {
            IO::write::<i32>(&mut *self, address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readi8v2(self: *mut Emu, address: u32) -> i32 {
        unsafe { IO::read_ext::<i8, ext::Sign>(&mut *self, address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readu8v2(self: *mut Emu, address: u32) -> u32 {
        unsafe { IO::read_ext::<u8, ext::Zero>(&mut *self, address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readi16v2(self: *mut Emu, address: u32) -> i32 {
        unsafe { IO::read_ext::<i16, ext::Sign>(&mut *self, address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readu16v2(self: *mut Emu, address: u32) -> u32 {
        unsafe { IO::read_ext::<u16, ext::Zero>(&mut *self, address) }
    }

    /// # Safety
    /// safety my ass
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn read32v2(self: *mut Emu, address: u32) -> i32 {
        unsafe { IO::read_ext::<i32, ext::NoExt>(&mut *self, address) }
    }

    pub unsafe extern "C" fn ext_run_io(self: *mut Emu) {
        unsafe {
            self.as_mut_unchecked().run_io();
        }
    }
}
