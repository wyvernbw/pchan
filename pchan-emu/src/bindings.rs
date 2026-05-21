use std::ops::{Deref, DerefMut, Receiver};
use std::ptr::NonNull;

use crate::Emu;

#[repr(transparent)]
#[derive(derive_more::Deref, derive_more::DerefMut)]
pub struct NonNullRecv<T>(NonNull<T>);

use crate::memory::ext;
impl Emu {
    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn urread32(&mut self, address: u32, overwrite: u32) -> u32 {
        unsafe { self.read32_unaligned_r(address, overwrite) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn ulread32(&mut self, address: u32, overwrite: u32) -> u32 {
        unsafe { self.read32_unaligned_l(address, overwrite) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn ulwrite32(&mut self, address: u32, value: u32) {
        unsafe {
            self.write32_unaligned_l(address, value as _);
        }
    }
    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn urwrite32(&mut self, address: u32, value: u32) {
        unsafe {
            self.write32_unaligned_r(address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn write8v2(&mut self, address: u32, value: i32) {
        unsafe {
            (*self).write::<i8>(address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn write16v2(&mut self, address: u32, value: i32) {
        unsafe {
            (*self).write::<i16>(address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn write32v2(&mut self, address: u32, value: i32) {
        unsafe {
            (*self).write::<i32>(address, value as _);
        }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readi8v2(&mut self, address: u32) -> i32 {
        unsafe { (*self).read_ext::<i8, ext::Sign>(address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readu8v2(&mut self, address: u32) -> u32 {
        unsafe { (*self).read_ext::<u8, ext::Zero>(address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readi16v2(&mut self, address: u32) -> i32 {
        unsafe { (*self).read_ext::<i16, ext::Sign>(address) }
    }

    /// # Safety
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn readu16v2(&mut self, address: u32) -> u32 {
        unsafe { (*self).read_ext::<u16, ext::Zero>(address) }
    }

    /// # Safety
    /// safety my ass
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn read32v2(&mut self, address: u32) -> i32 {
        unsafe { (*self).read_ext::<i32, ext::NoExt>(address) }
    }

    /// # Safety
    pub unsafe extern "C" fn ext_run_io(&mut self) {
        unsafe {
            self.run_io();
        }
    }
}
