use core::alloc::Allocator;

use arbitrary_int::prelude::*;
use bitbybit::bitenum;
use derive_more as d;
use pchan_encode::derive::Endecode;
use pchan_utils::hex;

use crate::Emu;
use crate::io::{CastIOFrom, CastIOInto, UnhandledIO};

use super::irq::Irq;

const TIMER_UPDATE_FREQUENCY: u64 = 1000;

#[derive(Debug, Clone, Default, Endecode)]
pub struct TimerState {
    clock:       u64,
    pub timer_0: Timer<0>,
    pub timer_1: Timer<1>,
    pub timer_2: Timer<2>,

    timer_2_fract: u16,
    hblanks:       u16,
    in_vblank:     bool,
    in_hblank:     bool,
}

#[derive(Debug, Clone, Default, Endecode)]
pub struct Timer<const N: usize> {
    value:      TimerCounterValue,
    target:     TimerTarget,
    mode:       TimerCounterMode,
    irq:        Irq,
    armed:      bool,
    overflowed: bool,
    hit_target: bool,

    internal_sync: bool,
}

#[bitbybit::bitfield(u32, debug)]
#[derive(Default, Endecode)]
#[must_use]
pub struct TimerCounterValue {
    #[bits(0..=15, rw)]
    value: u16,
}

#[derive(Debug, Clone, Copy, d::Deref, d::DerefMut, Default, Endecode)]
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
#[derive(Default, Endecode)]
#[must_use]
pub struct TimerCounterMode {
    #[bit(0, rw)]
    sync_on:      bool,
    #[bits(1..=2, rw)]
    sync_mode0_1: SyncMode0_1,
    #[bit(3, rw)]
    reset_mode:   TimerResetMode,

    #[bit(4, rw)]
    irq_on_target:   bool,
    #[bit(5, rw)]
    irq_on_overflow: bool,
    #[bit(6, rw)]
    irq_repeat_mode: TimerIrqRepeatMode,
    #[bit(7, rw)]
    irq_level_mode:  TimerIrqLevelMode,

    #[bit(8, rw)]
    source:    bool,
    #[bit(9, rw)]
    prescaler: bool,

    #[bit(10, rw)]
    irq:              IrqFlag,
    #[bit(11, rw)]
    reached_target:   bool,
    #[bit(12, rw)]
    reached_overflow: bool,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug, Endecode)]
enum SyncMode0_1 {
    PauseDuringBlank     = 0x0,
    ResetAtBlank         = 0x1,
    ResetAtBlankAndPause = 0x2,
    PauseUntilBlank      = 0x3,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug, Default, PartialEq, Eq)]
enum IrqFlag {
    Y = 0x0,
    #[default]
    N = 0x1,
}

impl IrqFlag {
    fn inverted(self) -> Self {
        match self {
            IrqFlag::Y => IrqFlag::N,
            IrqFlag::N => IrqFlag::Y,
        }
    }
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
pub enum TimerIrqLevelMode {
    Pulse  = 0x0,
    Toggle = 0x1,
}

pub struct AdvanceTimerSummary {
    timer_0_old: u16,
    timer_0_new: u16,
}

impl<A: Allocator> Emu<A> {
    pub(crate) fn timers_init(&mut self) {
        self.timers.timer_0.irq = Irq::Irq4Timer0;
        self.timers.timer_1.irq = Irq::Irq5Timer1;
        self.timers.timer_2.irq = Irq::Irq6Timer2;
    }

    pub fn timers_read<T: Copy>(&mut self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100 => Ok(self.timers.timer_0.read_value().io_from_u32()),
            0x1f801104 => Ok(self.timers.timer_0.read_mode().io_from_u32()),
            0x1f801108 => Ok(self.timers.timer_0.target.io_from_u32()),
            0x1f801110 => Ok(self.timers.timer_1.read_value().io_from_u32()),
            0x1f801114 => Ok(self.timers.timer_1.read_mode().io_from_u32()),
            0x1f801118 => Ok(self.timers.timer_1.target.io_from_u32()),
            0x1f801120 => Ok(self.timers.timer_2.read_value().io_from_u32()),
            0x1f801124 => Ok(self.timers.timer_2.read_mode().io_from_u32()),
            0x1f801128 => Ok(self.timers.timer_2.target.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
        // .inspect(|value| tracing::debug!(target: "timers", "r({})={}", hex(address), hex(*value)))
    }

    pub fn timers_write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801100 => {
                self.timers.timer_0.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            0x1f801104 => {
                self.timers
                    .timer_0
                    .write_mode(TimerCounterMode::new_with_raw_value(value.io_into_u32()));
                tracing::info!("timer 0 mode: {:#?}", self.timers.timer_1.mode);
            }
            0x1f801108 => {
                self.timers.timer_0.target =
                    TimerTarget(TimerCounterValue::new_with_raw_value(value.io_into_u32()));
            }
            0x1f801110 => {
                self.timers.timer_1.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            0x1f801114 => {
                self.timers
                    .timer_1
                    .write_mode(TimerCounterMode::new_with_raw_value(value.io_into_u32()));
                tracing::info!("timer 1 mode: {:#?}", self.timers.timer_1.mode);
            }
            0x1f801118 => {
                self.timers.timer_1.target =
                    TimerTarget(TimerCounterValue::new_with_raw_value(value.io_into_u32()));
            }
            0x1f801120 => {
                self.timers.timer_2.value =
                    TimerCounterValue::new_with_raw_value(value.io_into_u32());
            }
            0x1f801124 => {
                self.timers
                    .timer_2
                    .write_mode(TimerCounterMode::new_with_raw_value(value.io_into_u32()));
            }
            0x1f801128 => {
                self.timers.timer_2.target =
                    TimerTarget(TimerCounterValue::new_with_raw_value(value.io_into_u32()));
            }
            _ => return Err(UnhandledIO(address)),
        }
        // tracing::debug!(target: "timers", "r({})={}", hex(address), hex(value));
        Ok(())
    }

    pub fn run_timers(&mut self, cycles: u16) {
        self.timers.clock += u64::from(cycles);
        if self.timers.clock < TIMER_UPDATE_FREQUENCY {
            return;
        }
        self.timers.clock -= TIMER_UPDATE_FREQUENCY;
        let cycles = TIMER_UPDATE_FREQUENCY as u16;

        // FIXME: add dotclock source
        if self.timers.timer_0.mode.source() {
            todo!()
        }
        self.timers.timer_0.tick_by(cycles);

        if self.timers.timer_1.internal_sync {
            let d_clock = self.timers.timer_1.d_clock(self.timers.hblanks, cycles);
            let d_clock = match self.timers.timer_1.mode.sync_mode0_1() {
                SyncMode0_1::PauseDuringBlank => {
                    if self.gpu.dp.in_vblank() {
                        0
                    } else {
                        d_clock
                    }
                }
                SyncMode0_1::ResetAtBlank => {
                    if self.gpu.dp.in_vblank() && !self.timers.in_vblank {
                        self.timers.timer_1.reset_value();
                    }
                    d_clock
                }
                SyncMode0_1::ResetAtBlankAndPause => {
                    if self.gpu.dp.in_vblank() && !self.timers.in_vblank {
                        self.timers.timer_1.reset_value();
                    }
                    if self.gpu.dp.in_vblank() { d_clock } else { 0 }
                }
                SyncMode0_1::PauseUntilBlank => {
                    if self.gpu.dp.in_vblank() {
                        self.timers.timer_1.internal_sync = false;
                        0
                    } else {
                        d_clock
                    }
                }
            };
            self.timers.timer_1.tick_by(d_clock);
        } else {
            let d_clock = self.timers.timer_1.d_clock(self.timers.hblanks, cycles);
            self.timers.timer_1.tick_by(d_clock);
        }

        if self.timers.timer_2.mode.prescaler() {
            let d_clock = cycles + self.timers.timer_2_fract;
            self.timers.timer_2.tick_by(d_clock / 8);
            self.timers.timer_2_fract = d_clock % 8;
        } else {
            self.timers.timer_2.tick_by(cycles);
        }

        self.timers.hblanks = 0;

        let irq0 = self.timers.timer_0.trigger_updates();
        let irq1 = self.timers.timer_1.trigger_updates();
        let irq2 = self.timers.timer_2.trigger_updates();
        if irq0 {
            self.irq_trigger(self.timers.timer_0.irq);
        }
        if irq1 {
            self.irq_trigger(self.timers.timer_1.irq);
        }
        if irq2 {
            self.irq_trigger(self.timers.timer_2.irq);
        }
        self.timers.timer_0.clear_irqs();
        self.timers.timer_1.clear_irqs();
        self.timers.timer_2.clear_irqs();

        self.timers.in_vblank = self.gpu.dp.in_vblank();
        self.timers.in_hblank = self.gpu.dp.in_hblank();
    }
}

impl<const N: usize> Timer<N> {
    pub fn write_mode(&mut self, mode: TimerCounterMode) {
        self.mode.set_sync_on(mode.sync_on());
        self.mode.set_sync_mode0_1(mode.sync_mode0_1());
        self.mode.set_reset_mode(mode.reset_mode());
        self.mode.set_irq_on_target(mode.irq_on_target());
        self.mode.set_irq_on_overflow(mode.irq_on_overflow());
        self.mode.set_irq_repeat_mode(mode.irq_repeat_mode());
        self.mode.set_irq_level_mode(mode.irq_level_mode());
        self.mode.set_source(mode.source());
        self.mode.set_prescaler(mode.prescaler());
        self.mode.set_irq(IrqFlag::N);
        self.value = TimerCounterValue::new_with_raw_value(0);
        self.armed = true;
        self.internal_sync = self.mode.sync_on();
    }

    fn read_value(&mut self) -> TimerCounterValue {
        self.value
    }

    fn read_mode(&mut self) -> TimerCounterMode {
        self.mode.set_reached_target(false);
        self.mode.set_reached_overflow(false);
        self.mode
    }

    fn reset_value(&mut self) {
        self.value.set_value(0);
    }

    pub fn tick_by(&mut self, d_clock: u16) {
        let old_value = self.value.value();
        let target = self.target.value();
        let (value, overflowed) = self.value.value().overflowing_add(d_clock);

        let hit_target_and_overflowed = overflowed && (value >= target || old_value < target);
        let hit_target = (value >= target) && (old_value < target);
        if hit_target_and_overflowed || hit_target {
            self.hit_target = true;
        }
        self.value.set_value(value);
        self.overflowed = overflowed;
    }

    pub fn trigger_updates(&mut self) -> bool {
        if self.overflowed {
            self.mode.set_reached_overflow(true);
        }

        let mut irq_pending = false;
        if self.overflowed {
            if self.mode.irq_on_overflow() {
                irq_pending |= self.trigger_irq();
            }
            if self.mode.reset_mode() == TimerResetMode::OnOverflow {
                self.value.set_value(0);
            }
        }
        if self.hit_target {
            if self.mode.irq_on_target() {
                irq_pending |= self.trigger_irq();
            }
            if self.mode.reset_mode() == TimerResetMode::OnTarget {
                self.value.set_value(0);
            }
        }

        self.hit_target = false;
        self.overflowed = false;

        if self.mode.irq_repeat_mode() == TimerIrqRepeatMode::Repeat {
            self.armed = true;
        }

        irq_pending
    }

    /// consumes `armed`
    fn trigger_irq(&mut self) -> bool {
        if !self.armed {
            return false;
        }
        self.armed = false;
        match self.mode.irq_level_mode() {
            TimerIrqLevelMode::Pulse => {
                self.mode.set_irq(IrqFlag::Y);
                true
            }
            TimerIrqLevelMode::Toggle => {
                self.mode.set_irq(self.mode.irq().inverted());
                true
            }
        }
    }

    fn clear_irqs(&mut self) {
        if self.mode.irq_level_mode() == TimerIrqLevelMode::Pulse {
            self.hit_target = false;
            self.overflowed = false;
            self.mode.set_irq(IrqFlag::N);
        }
    }
}

impl Timer<1> {
    fn d_clock(&self, hblanks: u16, cpu_cycles: u16) -> u16 {
        match self.mode.source() {
            true => hblanks,
            false => cpu_cycles,
        }
    }
}

impl TimerState {
    pub fn trigger_hblank_start(&mut self) {}
    pub fn trigger_hblank_end(&mut self) {
        self.hblanks = self.hblanks.wrapping_add(1);
    }
}

impl TimerCounterMode {
    fn write(&mut self, value: TimerCounterMode) {
        *self = value;
        self.set_irq(IrqFlag::N);
    }
}

trait HasSyncMode {
    type SyncMode;
}

impl HasSyncMode for Timer<0> {
    type SyncMode = SyncMode0_1;
}

impl HasSyncMode for Timer<1> {
    type SyncMode = SyncMode0_1;
}

impl HasSyncMode for Timer<2> {
    type SyncMode = !;
}

impl<const N: usize> HasSyncMode for Timer<N> {
    default type SyncMode = !;
}
