use core::alloc::Allocator;

use crate::io::{CastIOFrom, IOResult, UnhandledIO};
use crate::{Emu, trace_todo};

impl<A: Allocator> Emu<A> {
    pub(crate) fn mdec_read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fff_ffff;
        match address {
            0x1f801820..=0x1f801824 => trace_todo!(0xffffffffu32, "read to mdec io register"),
            _ => Err(UnhandledIO(address)),
        }
    }
    pub(crate) fn mdec_write(&self, address: u32, value: u32) -> IOResult<()> {
        let address = address & 0x1fff_ffff;
        match address {
            0x1f801820..=0x1f801824 => trace_todo!("write to mdec io register"),
            _ => Err(UnhandledIO(address)),
        }
    }
}
