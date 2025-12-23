use std::ops::RangeInclusive;

use crate::{
    Bus, Emu,
    cpu::Exception,
    io::{IO, UnhandledIO},
    memory::{GUEST_MEM_MAP, MEM_MAP},
};
use bitfield::bitfield;

bitfield! {
    #[derive(Clone, Copy)]
    pub struct TimerCounterValue(u32);
    impl Debug;

    u16;
    value, set_value: 15, 0;
}

bitfield! {
    #[derive(Clone, Copy)]
    pub struct TimerCounterTarget(u32);
    impl Debug;

    u16;
    value, set_value: 15, 0;
}

bitfield! {
    #[derive(Clone, Copy)]
    pub struct TimerCounterMode(u32);
    impl Debug;

    sync_enabled, set_sync_enabled: 0;
    sync_mode, set_sync_mode: 2, 1;

    u8, reset_mode, set_reset_mode: 3, 3;

    irq_on_target, set_irq_on_target: 4;
    irq_on_overflow, set_irq_on_overflow: 5;
    irq_repeat_mode, set_irq_repeat_mode: 6;
    irq_toggle_mode, set_irq_toggle_mode: 7;
    clock_source, set_clock_source: 9, 8;
    irq, set_irq: 10;
    reached_target, set_reached_target: 11;
    reached_overflow, set_reached_overflow: 12;
}

#[repr(u8)]
pub enum TimerResetMode {
    OnOverflow = 0x0,
    OnTarget = 0x1,
}

pub enum TimerRepeatMode {
    Oneshot = 0x0,
    Repeat = 0x1,
}

pub enum TimerToggleMode {
    Pulse = 0x0,
    Toggle = 0x1,
}

pub enum TimerClockSource {
    SystemClock,
    Other,
}

pub struct AdvanceTimerSummary {
    timer_0_old: u16,
    timer_0_new: u16,
}

pub trait Timers: Bus + IO {
    fn read_timers<T: Copy>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100..=0x1f801108 | 0x1f801110..=0x1f801118 | 0x1f801120..=0x1f801128 => {
                tracing::trace!("read to timer registers");
                Ok(self
                    .mem()
                    .read_region(MEM_MAP.io, GUEST_MEM_MAP.io, address))
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn write_timers<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100..=0x1f801108 | 0x1f801110..=0x1f801118 | 0x1f801120..=0x1f801128 => {
                tracing::trace!("write to timer registers");
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

        self.read::<TimerCounterMode>(timer_address)
    }

    fn timer_counter_target(&self, timer: u8) -> TimerCounterTarget {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801108 + timer as u32 * 0x10;

        self.read::<TimerCounterTarget>(timer_address)
    }

    fn timer_counter_value(&self, timer: u8) -> TimerCounterValue {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801100 + timer as u32 * 0x10;

        self.read::<TimerCounterValue>(timer_address)
    }

    fn set_timer_counter_value(&mut self, timer: u8, value: TimerCounterValue) {
        debug_assert!((0..=4).contains(&timer));
        let timer_address = 0x1f801100 + timer as u32 * 0x10;
        self.write(timer_address, value);
    }

    fn set_timer_counter_mode(&mut self, timer: u8, value: TimerCounterMode) {
        debug_assert!((0..=4).contains(&timer));
        let timer_address = 0x1f801104 + timer as u32 * 0x10;
        self.write(timer_address, value);
    }

    fn advance_timers(&self) -> AdvanceTimerSummary {
        let timer_0_address = 0x1f801100;
        let timer_0 = self.read::<TimerCounterValue>(timer_0_address);
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

            if timer_mode.reset_mode() == TimerResetMode::OnTarget as u8 {
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
            self.cpu_mut().handle_exception(Exception::Interrupt);
        }
    }
}

impl Timers for Emu {}
