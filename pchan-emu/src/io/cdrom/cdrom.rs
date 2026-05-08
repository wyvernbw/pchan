use crate::{
    Bus, Emu,
    io::{CastIOFrom, UnhandledIO},
};

#[derive(Default, derive_more::Debug, Clone)]
pub struct CDRomState {
    bank: u8,
}

macro_rules! trace_todo {
    ($args: tt) => {{
        tracing::error!($args);
        Ok(())
    }};
    ($value: expr, $args: tt) => {{
        tracing::error!($args);
        Ok($value.io_from_u32())
    }};
}

/// Current todo:
/// ```log
/// ERROR pchan_emu::io::cdrom: todo(cdrom): write to status reg
/// ERROR pchan_emu::io::cdrom: todo(cdrom): write to request register
/// ERROR pchan_emu::io::cdrom: todo(cdrom): write to status reg
/// ERROR pchan_emu::io::cdrom: todo(cdrom): write to param fifo
/// ```
pub trait CDRom: Bus {
    fn write<T>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank;
        match (address, bank) {
            (0x1f801800, _) => trace_todo!("todo(cdrom): write to status reg"),

            (0x1f801801, 0) => trace_todo!("todo(cdrom): write to cd command register"),
            (0x1f801801, 1) => Ok(()), // unused
            (0x1f801801, 2) => Ok(()), // unused
            (0x1f801801, 3) => {
                trace_todo!(
                    "todo(cdrom): write to cd audio volume for right-cd-out to right-spu-in"
                )
            }

            (0x1f801802, 0) => trace_todo!("todo(cdrom): write to param fifo"),
            (0x1f801802, 1) => trace_todo!("todo(cdrom): write to irq on/off register"),
            (0x1f801802, 2) => {
                trace_todo!("todo(cdrom): write to cd audio volume for left-cd-out to left-spu-in")
            }
            (0x1f801802, 3) => {
                trace_todo!("todo(cdrom): write to cd audio volume for right-cd-out to left-spu-in")
            }

            (0x1f801803, 0) => trace_todo!("todo(cdrom): write to request register"),
            (0x1f801803, 1) => {
                trace_todo!("todo(cdrom): write to cd irq flag register")
            }
            (0x1f801803, 2) => {
                trace_todo!("todo(cdrom): write to cd audio volume for left-cd-out to right-spu-in")
            }
            (0x1f801803, 3) => {
                trace_todo!("todo(cdrom): write to cd audio volume apply")
            }
            _ => Err(UnhandledIO(address)),
        }
    }
    fn read<T>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank;
        match (address, bank) {
            (0x1f801801, _) => trace_todo!(0u32, "todo(cdrom): read from response fifo"),

            (0x1f801802, _) => trace_todo!(0u32, "todo(cdrom): read from data fifo"),

            (0x1f801803, 0 | 2) => {
                trace_todo!(0u32, "todo(cdrom): read from cd irq on/off register")
            }
            (0x1f801803, 1 | 3) => {
                trace_todo!(0u32, "todo(cdrom): read from cd irq flag register")
            }
            _ => Err(UnhandledIO(address)),
        }
    }
}

impl CDRom for Emu {}
