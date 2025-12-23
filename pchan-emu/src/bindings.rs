use crate::{Emu, io::IO};

use pchan_utils::hex;
use tracing::{Level, instrument};

use crate::memory::ext;
impl Emu {
    #[unsafe(no_mangle)]
    #[pchan_macros::instrument_write]
    pub unsafe extern "C" fn write8v2(self: *mut Emu, address: u32, value: i32) {
        tracing::trace!("write");
        unsafe {
            IO::write::<i8>(&mut *self, address, value as _);
        }
    }

    #[unsafe(no_mangle)]
    #[pchan_macros::instrument_write]
    pub unsafe extern "C" fn write16v2(self: *mut Emu, address: u32, value: i32) {
        tracing::trace!("write");
        unsafe {
            IO::write::<i16>(&mut *self, address, value as _);
        }
    }

    #[unsafe(no_mangle)]
    #[pchan_macros::instrument_write]
    pub unsafe extern "C" fn write32v2(self: *mut Emu, address: u32, value: i32) {
        tracing::trace!("write");
        unsafe {
            IO::write::<i32>(&mut *self, address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readi8v2(self: *mut Emu, address: u32) -> i32 {
        unsafe { IO::read_ext::<i8, ext::Sign>(&*self, address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readu8v2(self: *mut Emu, address: u32) -> u32 {
        unsafe { IO::read_ext::<u8, ext::Zero>(&*self, address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readi16v2(self: *mut Emu, address: u32) -> i32 {
        unsafe { IO::read_ext::<i16, ext::Sign>(&*self, address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readu16v2(self: *mut Emu, address: u32) -> u32 {
        unsafe { IO::read_ext::<u16, ext::Zero>(&*self, address) }
    }

    /// # Safety
    /// safety my ass
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn read32v2(self: *mut Emu, address: u32) -> i32 {
        unsafe { IO::read_ext::<i32, ext::NoExt>(&*self, address) }
    }
}
