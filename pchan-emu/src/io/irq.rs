use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use derive_more as d;
use pchan_utils::hex;

use crate::{
    Bus, Emu,
    cpu::exceptions::Exceptions,
    io::{CastIOFrom, CastIOInto, IO, IOResult, UnhandledIO},
};

#[derive(Debug, Clone, Copy, Hash, Default)]
pub struct IrqState {
    pub i_stat: IrqField,
    pub i_mask: IrqField,
}

#[bitfield(u32, debug)]
#[derive(d::Deref, Hash, Default)]
pub struct IrqField {
    #[bit(0)]
    irq0_vblank: bool,
    #[bit(1)]
    irq1_gpu:    bool,
    #[bit(2)]
    irq2_cdrom:  bool,
    #[bit(3)]
    irq3_dma:    bool,
    #[bit(4)]
    irq4_timer0: bool,
    #[bit(5)]
    irq5_timer1: bool,
    #[bit(6)]
    irq6_timer2: bool,
    #[bit(7)]
    irq7_sio0:   bool,

    #[bit(0, rw)]
    irq_flag:           [bool; 11],
    #[bits(0..=10, rw)]
    irq_flags_combined: u11,
}

#[bitenum(u8)]
#[derive(Debug, PartialEq, Eq, Default, strum::EnumIter)]
pub enum Irq {
    #[default]
    Irq0Vblank           = 0x0,
    Irq1Gpu              = 0x1,
    Irq2CDRom            = 0x2,
    Irq3Dma              = 0x3,
    Irq4Timer0           = 0x4,
    Irq5Timer1           = 0x5,
    Irq6Timer2           = 0x6,
    Irq7JoypadAndMemcard = 0x7,
}

impl IrqState {
    pub fn trigger_irq(&mut self, irq: Irq) {
        let old_stat = self.i_stat;
        self.i_stat.set_irq_flag(irq as usize, true);

        if irq != Irq::Irq0Vblank {
            let mask = self.i_mask;
            tracing::info!(
                ?irq,
                "{:010b} mask{:010b} -> {:010b}",
                old_stat.irq_flags_combined(),
                mask.irq_flags_combined(),
                self.i_stat.irq_flags_combined()
            );
        }
    }
}

pub trait Interrupts: Bus + IO + Exceptions {
    fn trigger_irq(&mut self, irq: Irq) {
        self.irq_mut().trigger_irq(irq);
        self.run_irq_io();
    }
    #[pchan_macros::instrument(
        level = "trace", "irq:r",
        skip_all,
        fields(address=%hex(address))
    )]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        match address {
            0x1f801070 => Ok(self.irq().i_stat.io_from_u32()),
            0x1f801074 => Ok(self.irq().i_mask.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[pchan_macros::instrument(
        level = "info",
        "irq:w",
        skip_all,
        fields(
            pc=%hex(self.cpu().pc),
            address=%hex(address),
            value=%hex(value.io_into_u32())
        )
    )]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        match address {
            0x1f801070 => {
                let irq = self.irq_mut();
                let flags = irq.i_stat.irq_flags_combined();
                let write = value.io_into_u32();
                let flags = flags & write.as_();
                irq.i_stat.set_irq_flags_combined(flags);
                tracing::trace!("{flags:010b}");
                if flags.as_u16().count_ones() == 0 {
                    self.clear_irq();
                }
                self.run_irq_io();

                Ok(())
            }
            0x1f801074 => {
                self.irq_mut().i_mask = IrqField::new_with_raw_value(
                    value.io_into_u32_overwrite(self.irq_mut().i_mask.raw_value()),
                );
                self.run_irq_io();
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn handle_ev_irq(&mut self, _: usize, _: u64) {
        self.run_irq_io();
    }

    fn run_irq_io(&mut self) {
        if self.irq().i_stat.irq_flags_combined().as_u32()
            & self.irq().i_mask.irq_flags_combined().as_u32()
            != 0
        {
            self.raise_irq_exception();
        }
    }
}

impl Interrupts for Emu {}
