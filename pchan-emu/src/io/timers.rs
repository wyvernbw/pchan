use arbitrary_int::prelude::*;
use derive_more as d;
use std::ops::RangeInclusive;

use crate::{
    Bus, Emu,
    cpu::exceptions::{Exception, Exceptions},
    io::{IO, UnhandledIO, irq::Interrupts},
    memory::{GUEST_MEM_MAP, MEM_MAP},
};
use bitfield::bitfield;

use super::irq::Irq;

#[bitbybit::bitfield(u32, debug)]
pub struct TimerCounterValue {
    #[bits(0..=15, rw)]
    value: u16,
}

#[derive(Debug, Clone, Copy, d::Deref, d::DerefMut)]
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
    fn read_timers<T: Copy>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100..=0x1f801108 | 0x1f801110..=0x1f801118 | 0x1f801120..=0x1f801128 => Ok(self
                .mem()
                .read_region(MEM_MAP.io, GUEST_MEM_MAP.io, address)),
            _ => Err(UnhandledIO(address)),
        }
    }

    fn write_timers<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100..=0x1f801108 | 0x1f801110..=0x1f801118 | 0x1f801120..=0x1f801128 => {
                self.mem_mut()
                    .write_region(MEM_MAP.io, GUEST_MEM_MAP.io, address, value);
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn run_timer_pipeline(&mut self) {
        let adv = self.advance_timers();
        self.trigger_timer_updates(adv);
    }

    fn timer_counter_mode(&self, timer: u8) -> TimerCounterMode {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801104 + timer as u32 * 0x10;

        self.read_timers::<TimerCounterMode>(timer_address).unwrap()
    }

    fn timer_counter_target(&self, timer: u8) -> TimerTarget {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801108 + timer as u32 * 0x10;

        self.read_timers::<TimerTarget>(timer_address).unwrap()
    }

    fn timer_counter_value(&self, timer: u8) -> TimerCounterValue {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801100 + timer as u32 * 0x10;

        self.read_timers::<TimerCounterValue>(timer_address)
            .unwrap()
    }

    fn set_timer_counter_value(&mut self, timer: u8, value: TimerCounterValue) {
        debug_assert!((0..=4).contains(&timer));
        let timer_address = 0x1f801100 + timer as u32 * 0x10;
        self.write_timers(timer_address, value).unwrap();
    }

    fn set_timer_counter_mode(&mut self, timer: u8, value: TimerCounterMode) {
        debug_assert!((0..=4).contains(&timer));
        let timer_address = 0x1f801104 + timer as u32 * 0x10;
        self.write_timers(timer_address, value).unwrap();
    }

    fn advance_timers(&self) -> AdvanceTimerSummary {
        let timer_0_address = 0x1f801100;
        let timer_0 = self
            .read_timers::<TimerCounterValue>(timer_0_address)
            .unwrap();
        let timer_0_value = timer_0.value();
        // timer 0 is synced to system clock
        let (new_timer_0_value, _overflowed) =
            timer_0_value.overflowing_add(self.cpu().d_clock as _);

        // TODO: timer 1 and 2

        AdvanceTimerSummary {
            timer_0_old: timer_0_value,
            timer_0_new: new_timer_0_value,
        }
    }

    fn check_target_range(
        mut timer_mode: TimerCounterMode,
        mut timer_value: TimerCounterValue,
        range: impl Into<RangeInclusive<u16>>,
        target: u16,
        new_value: u16,
    ) -> (TimerCounterMode, TimerCounterValue) {
        let range = range.into();
        if range.contains(&target) {
            timer_mode.set_reached_target(true);

            if timer_mode.irq_on_target() {
                timer_value.set_value(new_value);
                // irq
                timer_mode.set_irq(true);
            }

            if timer_mode.reset_mode() == TimerResetMode::OnTarget {
                timer_value.set_value(0);
            }
        }
        (timer_mode, timer_value)
    }

    fn trigger_timer_updates(&mut self, adv: AdvanceTimerSummary) {
        // timer 0
        let timer_0_mode = self.timer_counter_mode(0);
        let timer_0_target = self.timer_counter_target(0);
        let mut new_timer_0_mode = timer_0_mode;
        let mut new_timer_0_value = self.timer_counter_value(0);

        let overflowed = adv.timer_0_new < adv.timer_0_old;

        if overflowed {
            new_timer_0_value.set_value(adv.timer_0_new);
            new_timer_0_mode.set_reached_overflow(true);
            if timer_0_mode.irq_on_overflow() {
                new_timer_0_mode.set_irq(true);
            }
        }

        let target_value = timer_0_target.value();
        // if timer overflows, you have to check 2 ranges:
        // |0 ###### now ...... then ###### u32::MAX |
        // +----------->        |-------------------->
        //   range a                    range b
        let (new_timer_0_mode, new_timer_0_value) = match overflowed {
            true => {
                // check both ranges
                let (mode, value) = Self::check_target_range(
                    new_timer_0_mode,
                    new_timer_0_value,
                    adv.timer_0_old..=u16::MAX,
                    target_value,
                    adv.timer_0_new,
                );
                let (mode, value) = Self::check_target_range(
                    mode,
                    value,
                    0..=adv.timer_0_new,
                    target_value,
                    adv.timer_0_new,
                );
                (mode, value)
            }
            false => Self::check_target_range(
                new_timer_0_mode,
                new_timer_0_value,
                adv.timer_0_old..=adv.timer_0_new,
                target_value,
                adv.timer_0_new,
            ),
        };

        self.set_timer_counter_value(0, new_timer_0_value);
        self.set_timer_counter_mode(0, new_timer_0_mode);

        if new_timer_0_mode.irq() {
            self.trigger_irq(Irq::Irq4Timer0);
        }
    }
}

impl Timers for Emu {}
