use core::mem::offset_of;
use core::ptr::NonNull;

use crate::Emu;

#[repr(transparent)]
#[derive(derive_more::Deref, derive_more::DerefMut)]
pub struct NonNullRecv<T>(NonNull<T>);

use crate::cpu::gte::GteBindings;
use crate::memory::{ExtendExt, ext};
impl Emu {
    /// # Safety
    pub unsafe extern "C" fn urread32(self: *mut Self, address: u32, overwrite: u32) -> u32 {
        unsafe {
            self.as_mut_unchecked()
                .read32_unaligned_r(address, overwrite)
        }
    }

    /// # Safety
    pub unsafe extern "C" fn ulread32(self: *mut Self, address: u32, overwrite: u32) -> u32 {
        unsafe {
            self.as_mut_unchecked()
                .read32_unaligned_l(address, overwrite)
        }
    }

    /// # Safety
    pub unsafe extern "C" fn ulwrite32(self: *mut Self, address: u32, value: u32) {
        unsafe {
            self.as_mut_unchecked()
                .write32_unaligned_l(address, value as _);
        }
    }

    /// # Safety
    pub unsafe extern "C" fn urwrite32(self: *mut Self, address: u32, value: u32) {
        unsafe {
            self.as_mut_unchecked()
                .write32_unaligned_r(address, value as _);
        }
    }

    /// # Safety
    pub unsafe extern "C" fn write8v2(self: *mut Self, address: u32, value: i32) {
        unsafe {
            self.as_mut_unchecked().write::<i8>(address, value as u32);
        }
    }

    /// # Safety
    pub unsafe extern "C" fn write16v2(self: *mut Self, address: u32, value: i32) {
        unsafe {
            self.as_mut_unchecked().write::<i16>(address, value as u32);
        }
    }

    /// # Safety
    pub unsafe extern "C" fn write32v2(self: *mut Self, address: u32, value: i32) {
        unsafe {
            self.as_mut_unchecked().write::<i32>(address, value as u32);
        }
    }

    /// # Safety
    pub unsafe extern "C" fn readi8v2(self: *mut Self, address: u32) -> i32 {
        unsafe {
            self.as_mut_unchecked()
                .read::<i8>(address)
                .ext::<ext::Sign>()
        }
    }

    /// # Safety
    pub unsafe extern "C" fn readu8v2(self: *mut Self, address: u32) -> u32 {
        unsafe {
            self.as_mut_unchecked()
                .read::<u8>(address)
                .ext::<ext::Zero>()
        }
    }

    /// # Safety
    pub unsafe extern "C" fn readi16v2(self: *mut Self, address: u32) -> i32 {
        unsafe {
            self.as_mut_unchecked()
                .read::<i16>(address)
                .ext::<ext::Sign>()
        }
    }

    /// # Safety
    pub unsafe extern "C" fn readu16v2(self: *mut Self, address: u32) -> u32 {
        unsafe {
            self.as_mut_unchecked()
                .read::<u16>(address)
                .ext::<ext::Zero>()
        }
    }

    /// # Safety
    pub unsafe extern "C" fn read32v2(self: *mut Self, address: u32) -> i32 {
        unsafe { self.as_mut_unchecked().read::<i32>(address) }
    }

    /// # Safety
    pub unsafe extern "C" fn ext_run_io(self: *mut Self) {
        unsafe {
            smol::block_on(self.as_mut_unchecked().run_io());
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct ExternFnTable {
    write32v2:      unsafe extern "C" fn(*mut Emu, u32, i32),
    write16v2:      unsafe extern "C" fn(*mut Emu, u32, i32),
    write8v2:       unsafe extern "C" fn(*mut Emu, u32, i32),
    ulwrite32:      unsafe extern "C" fn(*mut Emu, u32, u32),
    urwrite32:      unsafe extern "C" fn(*mut Emu, u32, u32),
    readi8v2:       unsafe extern "C" fn(*mut Emu, u32) -> i32,
    readu8v2:       unsafe extern "C" fn(*mut Emu, u32) -> u32,
    readi16v2:      unsafe extern "C" fn(*mut Emu, u32) -> i32,
    readu16v2:      unsafe extern "C" fn(*mut Emu, u32) -> u32,
    read32v2:       unsafe extern "C" fn(*mut Emu, u32) -> i32,
    ulread32:       unsafe extern "C" fn(*mut Emu, u32, u32) -> u32,
    urread32:       unsafe extern "C" fn(*mut Emu, u32, u32) -> u32,
    handle_syscall: unsafe extern "C" fn(&mut Emu, pc: u32),
    handle_rfe:     unsafe extern "C" fn(&mut Emu),
    handle_break:   unsafe extern "C" fn(&mut Emu, pc: u32),
    cop0_set_cause: fn(&mut Emu, u32),
    pub(crate) gte: GteBindings,
}

impl ExternFnTable {
    pub const OFFSET: u32 = offset_of!(Emu, fn_table) as u32;
    pub const WRITE32V2: u32 =
        offset_of!(Self, write32v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const WRITE16V2: u32 =
        offset_of!(Self, write16v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const WRITE8V2: u32 = offset_of!(Self, write8v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const ULWRITE32: u32 =
        offset_of!(Self, ulwrite32) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const URWRITE32: u32 =
        offset_of!(Self, urwrite32) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const READI8V2: u32 = offset_of!(Self, readi8v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const READU8V2: u32 = offset_of!(Self, readu8v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const READI16V2: u32 =
        offset_of!(Self, readi16v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const READU16V2: u32 =
        offset_of!(Self, readu16v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const READ32V2: u32 = offset_of!(Self, read32v2) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const ULREAD32: u32 = offset_of!(Self, ulread32) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const URREAD32: u32 = offset_of!(Self, urread32) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const HANDLE_SYSCALL: u32 =
        offset_of!(Self, handle_syscall) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const HANDLE_RFE: u32 =
        offset_of!(Self, handle_rfe) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const HANDLE_BREAK: u32 =
        offset_of!(Self, handle_break) as u32 + offset_of!(Emu, fn_table) as u32;
    pub const COP0_SET_CAUSE: u32 =
        offset_of!(Self, cop0_set_cause) as u32 + offset_of!(Emu, fn_table) as u32;

    const CHECK: () = assert!(
        offset_of!(Emu, fn_table) + size_of::<Self>() < 32760,
        "immediate out of 32kb range"
    );

    pub(crate) const fn new() -> Self {
        Self {
            write32v2:      Emu::write32v2,
            write16v2:      Emu::write16v2,
            write8v2:       Emu::write8v2,
            ulwrite32:      Emu::ulwrite32,
            urwrite32:      Emu::urwrite32,
            readi8v2:       Emu::readi8v2,
            readu8v2:       Emu::readu8v2,
            readi16v2:      Emu::readi16v2,
            readu16v2:      Emu::readu16v2,
            read32v2:       Emu::read32v2,
            ulread32:       Emu::ulread32,
            urread32:       Emu::urread32,
            handle_syscall: Emu::handle_syscall,
            handle_rfe:     Emu::handle_rfe,
            handle_break:   Emu::handle_break,
            cop0_set_cause: Emu::cop0_set_cause,
            gte:            GteBindings::new(),
        }
    }
}

impl Default for ExternFnTable {
    fn default() -> Self {
        Self::new()
    }
}
