//! This module implements some wrappers around Mmap/MmapMut to also support a cheap "empty" variant.
// Unfortunately Memmap itself doesn't support a cheap zero-length variant

use core::ops::{Deref, DerefMut};

use crate::AssemblyOffset;
use mmap_rs::{MmapMut, MmapOptions, UnsafeMmapFlags};

/// A structure holding a buffer of executable memory. It also derefs to a `&[u8]`.
/// This structure does not allocate when its size is 0.
#[derive(Debug, Default)]
pub struct ExecutableBuffer {
    // length of the buffer that has actually been written to
    length: usize,
    // backing buffer
    buffer: Option<MmapMut>,
}

impl ExecutableBuffer {
    /// Obtain a pointer into the executable memory from an offset into it.
    /// When an offset returned from `DynasmLabelApi::offset` is used, the resulting pointer
    /// will point to the start of the first instruction after the offset call,
    /// which can then be jumped or called to divert control flow into the executable
    /// buffer. Note that if this buffer is accessed through an Executor, these pointers
    /// will only be valid as long as its lock is held. When no locks are held,
    /// the assembler is free to relocate the executable buffer when it requires
    /// more memory than available.
    ///
    /// The memory this pointer points to is owned by this `ExecutableBuffer`.
    /// The programmer is responsible for ensuring that pointers generated from this API do not
    /// outlive the `ExecutableBuffer`.
    pub fn ptr(&self, offset: AssemblyOffset) -> *const u8 {
        &raw const self[offset.0]
    }

    /// Create a new executable buffer, backed by a buffer of size `size`.
    /// It will start with an initialized length of 0.
    pub fn new(size: usize) -> Result<ExecutableBuffer, mmap_rs::Error> {
        let buffer = if size == 0 {
            None
        } else {
            let buf = unsafe {
                MmapOptions::new(size)?
                    .with_unsafe_flags(UnsafeMmapFlags::JIT)
                    .map_exec_mut()?
            };
            Some(buf)
        };

        Ok(ExecutableBuffer { length: 0, buffer })
    }

    /// Query the backing size of this executable buffer
    pub fn size(&self) -> usize {
        self.buffer.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    pub fn make_mut(&self) -> Result<(), mmap_rs::Error> {
        if let Some(_) = &self.buffer {
            unsafe {
                pthread_jit_write_protect_np(0);
            }
        }
        Ok(())
    }

    pub fn make_exec(&self, from: usize) -> Result<(), mmap_rs::Error> {
        if let Some(buf) = &self.buffer {
            unsafe {
                sys_icache_invalidate((buf.start() + from) as _, self.length.saturating_sub(from));
                pthread_jit_write_protect_np(1);
            }
        }
        Ok(())
    }

    pub fn set_len(&mut self, length: usize) {
        self.length = length;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    pub fn pthread_jit_write_protect_np(enabled: libc::c_int);
    pub fn sys_icache_invalidate(start: *const libc::c_void, len: usize);
}

impl Deref for ExecutableBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if let Some(map) = &self.buffer {
            &map[..self.length]
        } else {
            &[]
        }
    }
}
impl DerefMut for ExecutableBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if let Some(map) = &mut self.buffer {
            &mut map[..self.length]
        } else {
            &mut []
        }
    }
}
