use std::ops::{Range, RangeInclusive};

use crate::{
    cpu::{Cpu, Exception},
    io::IO,
    memory::{Memory, ext},
};
use bitfield::bitfield;
use tracing::instrument;

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

impl IO {
    pub fn run_timer_pipeline(cpu: &mut Cpu, mem: &mut Memory) {
        let adv = IO::advance_timers(cpu, mem);
        IO::trigger_timer_updates(cpu, mem, adv);
    }

    pub fn timer_counter_mode(cpu: &Cpu, mem: &Memory, timer: u8) -> TimerCounterMode {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801104 + timer as u32 * 0x10;

        mem.read::<TimerCounterMode, ext::NoExt>(cpu, timer_address)
    }

    pub fn timer_counter_target(cpu: &Cpu, mem: &Memory, timer: u8) -> TimerCounterTarget {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801108 + timer as u32 * 0x10;

        mem.read::<TimerCounterTarget, ext::NoExt>(cpu, timer_address)
    }

    pub fn timer_counter_value(cpu: &Cpu, mem: &Memory, timer: u8) -> TimerCounterValue {
        debug_assert!((0..=4).contains(&timer));

        let timer_address = 0x1f801100 + timer as u32 * 0x10;

        mem.read::<TimerCounterValue, ext::NoExt>(cpu, timer_address)
    }

    pub fn set_timer_counter_value(
        cpu: &Cpu,
        mem: &mut Memory,
        timer: u8,
        value: TimerCounterValue,
    ) {
        debug_assert!((0..=4).contains(&timer));
        let timer_address = 0x1f801100 + timer as u32 * 0x10;
        mem.write(cpu, timer_address, value);
    }

    pub fn set_timer_counter_mode(cpu: &Cpu, mem: &mut Memory, timer: u8, value: TimerCounterMode) {
        debug_assert!((0..=4).contains(&timer));
        let timer_address = 0x1f801104 + timer as u32 * 0x10;
        mem.write(cpu, timer_address, value);
    }

    pub fn advance_timers(cpu: &Cpu, mem: &Memory) -> AdvanceTimerSummary {
        let timer_0_address = 0x1f801100;
        let timer_0 = mem.read::<TimerCounterValue, ext::NoExt>(cpu, timer_0_address);
        let timer_0_value = timer_0.value();
        // timer 0 is synced to system clock
        let (new_timer_0_value, overflowed) = timer_0_value.overflowing_add(cpu.d_clock);

        // TODO: timer 1 and 2

        AdvanceTimerSummary {
            timer_0_old: timer_0_value,
            timer_0_new: new_timer_0_value,
        }
    }

    pub fn check_target_range(
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

    pub fn trigger_timer_updates(cpu: &mut Cpu, mem: &mut Memory, adv: AdvanceTimerSummary) {
        // timer 0
        let timer_0_mode = IO::timer_counter_mode(cpu, mem, 0);
        let timer_0_target = IO::timer_counter_target(cpu, mem, 0);
        let mut new_timer_0_mode = timer_0_mode;
        let mut new_timer_0_value = IO::timer_counter_value(cpu, mem, 0);

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
                let (mode, value) = IO::check_target_range(
                    new_timer_0_mode,
                    new_timer_0_value,
                    adv.timer_0_old..=u16::MAX,
                    target_value,
                    adv.timer_0_new,
                );
                let (mode, value) = IO::check_target_range(
                    mode,
                    value,
                    0..=adv.timer_0_new,
                    target_value,
                    adv.timer_0_new,
                );
                (mode, value)
            }
            false => IO::check_target_range(
                new_timer_0_mode,
                new_timer_0_value,
                adv.timer_0_old..=adv.timer_0_new,
                target_value,
                adv.timer_0_new,
            ),
        };

        IO::set_timer_counter_value(cpu, mem, 0, new_timer_0_value);
        IO::set_timer_counter_mode(cpu, mem, 0, new_timer_0_mode);

        if new_timer_0_mode.irq() {
            cpu.handle_exception(Exception::Interrupt);
        }
    }
}
