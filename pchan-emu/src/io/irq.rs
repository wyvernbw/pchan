use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use derive_more as d;
use pchan_macros::{pchan_instrument_read, pchan_instrument_write};

use crate::{
    Bus, Emu,
    cpu::exceptions::{Exception, Exceptions},
    io::{CastIOFrom, CastIOInto, IO, IOResult, UnhandledIO},
};

#[derive(Debug, Clone, Copy, Hash, Default)]
pub struct IrqState {
    i_stat: IrqField,
    i_mask: IrqField,
}

#[bitfield(u32, debug)]
#[derive(d::Deref, Hash, Default)]
pub struct IrqField {
    #[bit(0)]
    irq0_vblank: bool,
    #[bit(3)]
    irq3_dma:    bool,
    #[bit(4)]
    irq4_timer0: bool,
    #[bit(5)]
    irq5_timer1: bool,
    #[bit(6)]
    irq6_timer2: bool,

    #[bit(0, rw)]
    irq_flag:           [bool; 11],
    #[bits(0..=10, r)]
    irq_flags_combined: u11,
}

#[bitenum(u8)]
#[derive(Debug, PartialEq, Eq, Default)]
pub enum Irq {
    #[default]
    Irq0Vblank = 0x0,
    Irq1Gpu    = 0x1,
    Irq3Dma    = 0x3,
    Irq4Timer0 = 0x4,
    Irq5Timer1 = 0x5,
    Irq6Timer2 = 0x6,
}

pub trait Interrupts: Bus + IO + Exceptions {
    fn irq_mut(&mut self) -> &mut IrqState {
        &mut self.cpu_mut().irq
    }
    fn irq(&self) -> &IrqState {
        &self.cpu().irq
    }

    fn trigger_irq(&mut self, irq: Irq) {
        let stat = self.irq().i_stat;
        let mask = self.irq().i_mask;

        // irq flag is set even when masked!
        let new_stat = IrqField::new_with_raw_value(*stat | (1 << irq as u8));
        self.irq_mut().i_stat = new_stat;

        if irq != Irq::Irq0Vblank {
            tracing::trace!(
                ?irq,
                "{:010b} mask{:010b} -> {:010b}",
                stat.irq_flags_combined(),
                mask.irq_flags_combined(),
                new_stat.irq_flags_combined()
            );
        }
    }
    #[pchan_instrument_read("irq:r")]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        match address {
            0x1f801070 => Ok(self.irq().i_stat.io_from_u32()),
            0x1f801074 => Ok(self.irq().i_mask.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[pchan_instrument_write("irq:w")]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        match address {
            0x1f801070 => {
                let irq = self.irq_mut();
                let i_stat = irq.i_stat.raw_value();
                let write = value.io_into_u32_overwrite(i_stat);
                let i_stat = i_stat & write;
                irq.i_stat = IrqField::new_with_raw_value(i_stat);

                Ok(())
            }
            0x1f801074 => {
                self.irq_mut().i_mask = IrqField::new_with_raw_value(
                    value.io_into_u32_overwrite(self.irq_mut().i_mask.raw_value()),
                );
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn run_irq_io(&mut self) {
        if self.irq().i_stat.irq_flags_combined().as_u32()
            & self.irq().i_mask.irq_flags_combined().as_u32()
            != 0
        {
            self.raise_exception(Exception::Interrupt);
        } else {
            self.clear_exception();
        }
    }
}

impl Interrupts for Emu {}
