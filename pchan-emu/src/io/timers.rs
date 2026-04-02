use arbitrary_int::prelude::*;
use derive_more as d;
use std::ops::RangeInclusive;

use crate::{
    Bus, Emu,
    io::{CastIOFrom, CastIOInto, IO, UnhandledIO, irq::Interrupts},
    memory::{GUEST_MEM_MAP, MEM_MAP},
};

use super::irq::Irq;

#[derive(Debug, Clone, Default)]
pub struct TimerState {
    pub timer_0: Timer,
    pub timer_1: Timer,
    pub timer_2: Timer,

    timer_2_fract: u16,
}

#[derive(Debug, Clone, Default)]
pub struct Timer {
    value:      TimerCounterValue,
    target:     TimerTarget,
    mode:       TimerCounterMode,
    irq:        Irq,
    overflowed: bool,
    hit_target: bool,
}

#[bitbybit::bitfield(u32, debug)]
#[derive(Default)]
pub struct TimerCounterValue {
    #[bits(0..=15, rw)]
    value: u16,
}

#[derive(Debug, Clone, Copy, d::Deref, d::DerefMut, Default)]
pub struct TimerTarget(TimerCounterValue);

///```md
///   0     Synchronization Enable (0=Free Run, 1=Synchronize via Bit1-2)
///  1-2   Synchronization Mode   (0-3, see lists below)
///         Synchronization Modes for Counter 0:
///           0 = Pause counter during Hblank(s)
///           1 = Reset counter to 0000h at Hblank(s)
///           2 = Reset counter to 0000h at Hblank(s) and pause outside of Hblank
///           3 = Pause until Hblank occurs once, then switch to Free Run
///         Synchronization Modes for Counter 1:
///           Same as above, but using Vblank instead of Hblank
///         Synchronization Modes for Counter 2:
///           0 or 3 = Stop counter at current value (forever, no h/v-blank start)
///           1 or 2 = Free Run (same as when Synchronization Disabled)
///  3     Reset counter to 0000h  (0=After Counter=FFFFh, 1=After Counter=Target)
///  4     IRQ when Counter=Target (0=Disable, 1=Enable)
///  5     IRQ when Counter=FFFFh  (0=Disable, 1=Enable)
///  6     IRQ Once/Repeat Mode    (0=One-shot, 1=Repeatedly)
///  7     IRQ Pulse/Toggle Mode   (0=Short Bit10=0 Pulse, 1=Toggle Bit10 on/off)
///  8-9   Clock Source (0-3, see list below)
///         Counter 0:  0 or 2 = System Clock,  1 or 3 = Dotclock
///         Counter 1:  0 or 2 = System Clock,  1 or 3 = Hblank
///         Counter 2:  0 or 1 = System Clock,  2 or 3 = System Clock/8
///  10    Interrupt Request       (0=Yes, 1=No) (Set after Writing)    (W=1) (R)
///  11    Reached Target Value    (0=No, 1=Yes) (Reset after Reading)        (R)
///  12    Reached FFFFh Value     (0=No, 1=Yes) (Reset after Reading)        (R)
///  13-15 Unknown (seems to be always zero)
///  16-31 Garbage (next opcode)
/// ```
#[bitbybit::bitfield(u32, debug)]
#[derive(Default)]
pub struct TimerCounterMode {
    #[bit(0, rw)]
    sync_on:    bool,
    #[bits(1..=2, rw)]
    sync_mode:  u2,
    #[bit(3, rw)]
    reset_mode: TimerResetMode,

    #[bit(4, rw)]
    irq_on_target:   bool,
    #[bit(5, rw)]
    irq_on_overflow: bool,
    #[bit(6, rw)]
    irq_repeat_mode: TimerIrqRepeatMode,
    #[bit(7, rw)]
    irq_toggle_mode: TimerIrqRepeatMode,

    #[bits(8..=9, rw)]
    source: u2,

    #[bit(10, rw)]
    irq:              bool,
    #[bit(11, rw)]
    reached_target:   bool,
    #[bit(12, rw)]
    reached_overflow: bool,
}

#[bitbybit::bitenum(u1, exhaustive = true)]
#[derive(Debug, PartialEq, Eq)]
pub enum TimerResetMode {
    OnOverflow = 0x0,
    OnTarget   = 0x1,
}

#[bitbybit::bitenum(u1, exhaustive = true)]
#[derive(Debug, PartialEq, Eq)]
pub enum TimerIrqRepeatMode {
    Oneshot = 0x0,
    Repeat  = 0x1,
}

#[bitbybit::bitenum(u1, exhaustive = true)]
#[derive(Debug, PartialEq, Eq)]
pub enum TimerIrqToggleMode {
    Pulse  = 0x0,
    Toggle = 0x1,
}

pub struct AdvanceTimerSummary {
    timer_0_old: u16,
    timer_0_new: u16,
}

pub trait Timers: Bus + IO + Interrupts {
    fn init_timers(&mut self) {
        let timers = self.timers_mut();
        timers.timer_0.irq = Irq::Irq4Timer0;
        timers.timer_1.irq = Irq::Irq5Timer1;
        timers.timer_2.irq = Irq::Irq6Timer2;
    }
    fn read_timers<T: Copy>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100 => Ok(self.timers().timer_0.value.io_from_u32()),
            0x1f801104 => Ok(self.timers().timer_0.mode.io_from_u32()),
            0x1f801108 => Ok(self.timers().timer_0.target.io_from_u32()),
            0x1f801110 => Ok(self.timers().timer_1.value.io_from_u32()),
            0x1f801114 => Ok(self.timers().timer_1.mode.io_from_u32()),
            0x1f801118 => Ok(self.timers().timer_1.target.io_from_u32()),
            0x1f801120 => Ok(self.timers().timer_2.value.io_from_u32()),
            0x1f801124 => Ok(self.timers().timer_2.mode.io_from_u32()),
            0x1f801128 => Ok(self.timers().timer_2.target.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }

    fn write_timers<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100 => {
                self.timers_mut().timer_0.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            0x1f801104 => {
                self.timers_mut().timer_0.mode =
                    TimerCounterMode::new_with_raw_value(value.io_into_u32());
            }
            0x1f801108 => {
                self.timers_mut().timer_0.target =
                    TimerTarget(TimerCounterValue::new_with_raw_value(value.io_into_u32()));
            }
            0x1f801110 => {
                self.timers_mut().timer_1.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            0x1f801114 => {
                self.timers_mut().timer_1.mode =
                    TimerCounterMode::new_with_raw_value(value.io_into_u32());
            }
            0x1f801118 => {
                self.timers_mut().timer_1.target =
                    TimerTarget(TimerCounterValue::new_with_raw_value(value.io_into_u32()));
            }
            0x1f801120 => {
                self.timers_mut().timer_2.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            0x1f801124 => {
                self.timers_mut().timer_2.mode =
                    TimerCounterMode::new_with_raw_value(value.io_into_u32());
            }
            0x1f801128 => {
                self.timers_mut().timer_2.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            _ => return Err(UnhandledIO(address)),
        }
        Ok(())
    }

    fn run_timer_pipeline(&mut self) {
        let timers = self.timers_mut();
        if timers.timer_0.mode.irq() {
            let irq = timers.timer_0.irq;
            self.trigger_irq(irq);
        }
        let timers = self.timers_mut();
        if timers.timer_1.mode.irq() {
            let irq = timers.timer_1.irq;
            self.trigger_irq(irq);
        }
        let timers = self.timers_mut();
        if timers.timer_2.mode.irq() {
            let irq = timers.timer_2.irq;
            self.trigger_irq(irq);
        }
    }

    fn timers_advance_by_cpu(&mut self, cycles: u16) {
        let timers = self.timers_mut();
        if timers.timer_0.check_source([0x0, 0x2]) {
            timers.timer_0.tick_by(cycles);
        }
        if timers.timer_1.check_source([0x0, 0x2]) {
            timers.timer_1.tick_by(cycles);
        }
        if timers.timer_2.check_source([0x0, 0x1]) {
            timers.timer_2.tick_by(cycles);
        } else {
            timers.timer_2.tick_by(cycles / 8 + timers.timer_2_fract);
            timers.timer_2_fract = cycles % 8;
        }
    }
}

impl Timers for Emu {}

impl Timer {
    pub fn tick_by(&mut self, d_clock: u16) {
        let (value, overflowed) = self.value.value().overflowing_add(d_clock);
        if overflowed && self.target.value() > self.value.value() {
            self.hit_target = true;
        }
        self.value.set_value(value);
        self.overflowed = overflowed;
    }

    pub fn trigger_updates(&mut self) {
        if self.overflowed {
            self.mode.set_reached_overflow(true);
            if self.mode.irq_on_overflow() {
                self.mode.set_irq(true);
            }
        }

        if self.mode.irq_on_overflow() && self.overflowed {
            self.mode.set_irq(true);
            if self.mode.reset_mode() == TimerResetMode::OnOverflow {
                self.value.set_value(0);
            }
        }
        if self.mode.irq_on_target() && self.hit_target {
            self.mode.set_irq(true);
            if self.mode.reset_mode() == TimerResetMode::OnTarget {
                self.value.set_value(0);
            }
        }
    }

    pub fn check_source(&self, flags: [u8; 2]) -> bool {
        let source = self.mode.source().as_u8();
        flags.contains(&source)
    }
}

impl TimerState {
    pub fn trigger_hblank(&mut self) {
        if self.timer_1.check_source([1, 3]) {
            self.timer_1.tick_by(1);
        }
    }
}
