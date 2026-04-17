use std::{
    ops::Range,
    simd::{
        Select, Simd,
        cmp::{SimdPartialEq, SimdPartialOrd},
    },
};

use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use pchan_utils::MAX_SIMD_WIDTH;

/// # `1f801c08h+N*10h` Voice 0..23 Attack/Decay/Sustain/Release (ADSR) (32bit)
///
/// ```plaintext
///   ____lower 16bit (at 1F801C08h+N*10h)___________________________________
///   15    Attack Mode       (0=Linear, 1=Exponential)
///   -     Attack Direction  (Fixed, always Increase) (until Level 7FFFh)
///   14-10 Attack Shift      (0..1Fh = Fast..Slow)
///   9-8   Attack Step       (0..3 = "+7,+6,+5,+4")
///   -     Decay Mode        (Fixed, always Exponential)
///   -     Decay Direction   (Fixed, always Decrease) (until Sustain Level)
///   7-4   Decay Shift       (0..0Fh = Fast..Slow)
///   -     Decay Step        (Fixed, always "-8")
///   3-0   Sustain Level     (0..0Fh)  ;Level=(N+1)*800h
///   ____upper 16bit (at 1F801C0Ah+N*10h)___________________________________
///   31    Sustain Mode      (0=Linear, 1=Exponential)
///   30    Sustain Direction (0=Increase, 1=Decrease) (until Key OFF flag)
///   29    Not used?         (should be zero)
///   28-24 Sustain Shift     (0..1Fh = Fast..Slow)
///   23-22 Sustain Step      (0..3 = "+7,+6,+5,+4" or "-8,-7,-6,-5") (inc/dec)
///   21    Release Mode      (0=Linear, 1=Exponential)
///   -     Release Direction (Fixed, always Decrease) (until Level 0000h)
///   20-16 Release Shift     (0..1Fh = Fast..Slow)
///   -     Release Step      (Fixed, always "-8")
/// ```
#[bitfield(u32, debug)]
#[derive(Default)]
pub struct ADSRRegister {
    // accessors for IO
    /// at `0x1f801c08+N*0x10`
    #[bits(0..=15, rw)]
    lower: u16,
    /// at `0x1f801c0a+N*0x10`
    #[bits(16..=31, rw)]
    upper: u16,

    // sustain params
    /// 3-0 Sustain Level     (0..0Fh)  ;Level=(N+1)*800h
    #[bits(0..=3, rw)]
    sustain_lvl:   u4,
    #[bit(31, rw)]
    sustain_mode:  EasingMode,
    /// 30 Sustain Direction (0=Increase, 1=Decrease) (until Key OFF flag)
    #[bit(30, rw)]
    sustain_dir:   Direction,
    /// 28-24 Sustain Shift     (0..1Fh = Fast..Slow)
    #[bits(24..=28, rw)]
    sustain_shift: u5,
    /// 23-22 Sustain Step      (0..3 = "+7,+6,+5,+4" or "-8,-7,-6,-5") (inc/dec)
    #[bits(22..=23, rw)]
    sustain_step:  u2,

    // decay params
    #[bits(4..=7, rw)]
    decay_shift: u4,

    // attack params
    #[bits(8..=9, rw)]
    attack_step:  u2,
    #[bits(10..=14, rw)]
    attack_shift: u5,
    #[bit(15, rw)]
    attack_mode:  EasingMode,

    // release params
    /// 20-16 Release Shift     (0..1Fh = Fast..Slow)
    #[bits(16..=20, rw)]
    release_shift: u5,
    #[bit(21, rw)]
    release_mode:  EasingMode,
}

#[bitenum(u1, exhaustive = true)]
#[derive(derive_more::Debug, PartialEq, Eq)]
enum EasingMode {
    Linear = 0x0,
    Expo   = 0x1,
}

#[bitenum(u1, exhaustive = true)]
#[derive(derive_more::Debug, PartialEq, Eq)]
enum Direction {
    Inc = 0x0,
    Dec = 0x1,
}

impl ADSRRegister {
    const fn decay_step(&self) -> i8 {
        -8
    }
    const fn decay_dir(&self) -> Direction {
        Direction::Dec
    }
    const fn decay_mode(&self) -> EasingMode {
        EasingMode::Expo
    }
    const fn attack_dir(&self) -> Direction {
        Direction::Inc
    }
    /// Release Step (Fixed, always "-8")
    const fn release_step(&self) -> i8 {
        -8
    }
    /// Release Direction (Fixed, always Decrease) (until Level 0000h)
    const fn release_dir(&self) -> Direction {
        Direction::Dec
    }
}

#[bitenum(u1, exhaustive = true)]
#[derive(Default, Debug)]
enum VolumeMode {
    #[default]
    Fixed = 0x0,
    Sweep = 0x1,
}

#[bitfield(u16, default = 0x0, debug)]
pub struct VolumeRegister {
    #[bit(15, rw)]
    mode: VolumeMode,
}

#[bitfield(u16)]
pub struct FixedVolumeRegister {
    /// Voice volume/2    (-4000h..+3FFFh = Volume -8000h..+7FFEh)
    #[bits(0..=14, rw)]
    volume: i15,
}

#[bitenum(u1, exhaustive = true)]
pub enum Phase {
    Positive = 0x0,
    Negative = 0x1,
}

#[bitfield(u16)]
pub struct SweepVolumeRegister {
    #[bit(14, rw)]
    sweep_mode:  EasingMode,
    #[bit(13, rw)]
    sweep_dir:   Direction,
    #[bit(12, rw)]
    sweep_phase: Phase,
    #[bits(2..=6, rw)]
    sweep_shift: u5,
    /// 1-0 Sweep Step (0..3 = "+7,+6,+5,+4" or "-8,-7,-6,-5") (inc/dec)
    #[bits(0..=1, rw)]
    sweep_step:  u2,
}

#[derive(Clone, Copy)]
pub enum TypedVolumeRegister {
    Fixed(FixedVolumeRegister),
    Sweep(SweepVolumeRegister),
}

impl VolumeRegister {
    pub fn typed(self) -> TypedVolumeRegister {
        match self.mode() {
            VolumeMode::Fixed => TypedVolumeRegister::Fixed(
                FixedVolumeRegister::new_with_raw_value(self.raw_value()),
            ),
            VolumeMode::Sweep => TypedVolumeRegister::Sweep(
                SweepVolumeRegister::new_with_raw_value(self.raw_value()),
            ),
        }
    }
    pub fn as_fixed(self) -> Option<FixedVolumeRegister> {
        match self.typed() {
            TypedVolumeRegister::Fixed(fixed_volume_register) => Some(fixed_volume_register),
            TypedVolumeRegister::Sweep(_) => None,
        }
    }
}

#[repr(align(64))]
#[derive(derive_more::Debug, Clone)]
pub struct VolumeState {
    pub registers:     [VolumeRegister; 24],
    pub queued_vol:    [i16; 24],
    pub internal:      [i16; 24],
    pub sweep_counter: [u32; 24],
}

impl Default for VolumeState {
    fn default() -> Self {
        Self {
            registers:     Default::default(),
            queued_vol:    [Self::QUEUE_NONE; 24],
            internal:      Default::default(),
            sweep_counter: [ADSRState::ENVELOPE_COUNTER_MAX; 24],
        }
    }
}

impl VolumeState {
    // using this allows for easy simd
    const QUEUE_NONE: i16 = i16::MAX;

    pub fn set_register(&mut self, idx: usize, value: u16) {
        self.registers[idx] = VolumeRegister::new_with_raw_value(value);
        if let Some(fixed) = self.registers[idx].as_fixed() {
            self.queued_vol[idx] = fixed.volume().as_i16() * 2;
        } else {
            self.queued_vol[idx] = Self::QUEUE_NONE;
        }
    }

    pub fn clock(&mut self) {
        const N: usize = MAX_SIMD_WIDTH / 2;
        const CHUNKS: usize = 24 / N;
        const REM: usize = 24 % N;

        let none = Simd::splat(Self::QUEUE_NONE);
        for ch in 0..CHUNKS {
            let base = ch * N;
            let queued = Simd::<i16, 8>::from_slice(&self.queued_vol[base..base + 8]);
            let internal = Simd::<i16, 8>::from_slice(&self.internal[base..base + 8]);

            let updated = queued.simd_ne(none).select(queued, internal);

            updated.copy_to_slice(&mut self.internal[base..base + 8]);
            none.copy_to_slice(&mut self.queued_vol[base..base + 8]);
        }

        #[allow(clippy::reversed_empty_ranges)]
        for i in CHUNKS * N..24 {
            let q = self.queued_vol[i];
            if q != Self::QUEUE_NONE {
                self.internal[i] = q;
                self.queued_vol[i] = Self::QUEUE_NONE;
            }
        }

        for i in 0..24 {
            let TypedVolumeRegister::Sweep(sweep) = self.registers[i].typed() else {
                continue;
            };
            let params = EnvelopePhaseParams {
                dir:   sweep.sweep_dir(),
                mode:  sweep.sweep_mode(),
                shift: sweep.sweep_shift().into(),
                step:  sweep.sweep_step().as_u8() as u16 as i16,
            };
            let mut counter_decrement =
                ADSRState::ENVELOPE_COUNTER_MAX >> params.shift.saturating_sub(11);

            if params.dir == Direction::Inc
                && params.mode == EasingMode::Expo
                && self.internal[i] > 0x6000
            {
                counter_decrement >>= 2;
            }

            if self.sweep_counter[i] < counter_decrement {
                self.sweep_counter[i] = ADSRState::ENVELOPE_COUNTER_MAX;

                // update envelope

                let mut step = (7 - params.step) as i32;
                if params.dir == Direction::Dec {
                    step = !step;
                }

                let mut step = step << 11u16.saturating_sub(params.shift);

                let current_level = self.internal[i] as i32;
                if params.dir == Direction::Dec && params.mode == EasingMode::Expo {
                    step = (step * current_level) >> 15;
                }

                self.internal[i] = (current_level + step).clamp(0, 0x7FFF) as i16;
            } else {
                self.sweep_counter[i] -= counter_decrement;
            }
        }
    }
}

#[derive(derive_more::Debug, Default, Clone)]
pub struct ADSRState {
    pub adsr:        [ADSRRegister; 24],
    pub envelopes:   EnvelopeState,
    pub voice_left:  VolumeState,
    pub voice_right: VolumeState,
}

#[derive(derive_more::Debug, Default, Clone)]
#[repr(align(64))]
pub struct EnvelopeState {
    pub level:         [i16; 24],
    pub sustain_level: [i16; 24],
    pub phase:         [EnvelopePhase; 24],
    pub counter:       [u32; 24],
}

#[derive(derive_more::Debug, Default, Clone, PartialEq, Eq)]
#[repr(u16)]
pub enum EnvelopePhase {
    Attack,
    Decay,
    Sustain,
    #[default]
    Release,
}

#[inline(always)]
pub fn apply_volume(sample: i16, volume: i16) -> i16 {
    (((sample as i32) * (volume as i32)) >> 15) as i16
}

struct EnvelopePhaseParams {
    dir:   Direction,
    mode:  EasingMode,
    shift: u16,
    step:  i16,
}

impl ADSRState {
    const ENVELOPE_COUNTER_MAX: u32 = 1 << (33 - 11);

    pub fn key_on(&mut self, idx: usize) {
        self.envelopes.level[idx] = 0;
        self.envelopes.phase[idx] = EnvelopePhase::Attack;
    }

    pub fn key_off(&mut self, idx: usize) {
        self.envelopes.phase[idx] = EnvelopePhase::Release;
    }

    pub fn set_register(&mut self, idx: usize, nibble: usize, value: u16) {
        match nibble {
            0 => {
                self.adsr[idx].set_lower(value);
            }
            1 => {
                self.adsr[idx].set_upper(value);
            }
            _ => panic!("invalid nibble {nibble}"),
        }
        self.envelopes.sustain_level[idx] = self.adsr[idx].sustain_lvl().as_i16() * 0x800;
    }

    pub fn clock(&mut self) {
        self.phase_transitions();
        self.voice_left.clock();
        self.voice_right.clock();
        // TODO: try rayon
        for i in 0..24 {
            let params = self.phase_params(i);

            let mut counter_decrement =
                Self::ENVELOPE_COUNTER_MAX >> params.shift.saturating_sub(11);

            if params.dir == Direction::Inc
                && params.mode == EasingMode::Expo
                && self.envelopes.level[i] > 0x6000
            {
                counter_decrement >>= 2;
            }

            if self.envelopes.counter[i] < counter_decrement {
                self.envelopes.counter[i] = Self::ENVELOPE_COUNTER_MAX;

                // update envelope

                let mut step = (7 - params.step) as i32;
                if params.dir == Direction::Dec {
                    step = !step;
                }

                let mut step = step << 11u16.saturating_sub(params.shift);

                let current_level = self.envelopes.level[i] as i32;
                if params.dir == Direction::Dec && params.mode == EasingMode::Expo {
                    step = (step * current_level) >> 15;
                }

                self.envelopes.level[i] = (current_level + step).clamp(0, 0x7FFF) as i16;
            } else {
                self.envelopes.counter[i] -= counter_decrement;
            }
        }
    }

    fn phase_params(&self, idx: usize) -> EnvelopePhaseParams {
        let adsr = self.adsr[idx];
        match self.envelopes.phase[idx] {
            EnvelopePhase::Attack => EnvelopePhaseParams {
                dir:   adsr.attack_dir(),
                mode:  adsr.attack_mode(),
                shift: adsr.attack_shift().as_u16() as _,
                step:  adsr.attack_step().as_u16() as _,
            },
            EnvelopePhase::Decay => EnvelopePhaseParams {
                dir:   adsr.decay_dir(),
                mode:  adsr.decay_mode(),
                shift: adsr.decay_shift().as_u16() as _,
                step:  adsr.decay_step().as_u16() as _,
            },
            EnvelopePhase::Sustain => EnvelopePhaseParams {
                dir:   adsr.sustain_dir(),
                mode:  adsr.sustain_mode(),
                shift: adsr.sustain_shift().as_u16() as _,
                step:  adsr.sustain_step().as_u16() as _,
            },
            EnvelopePhase::Release => EnvelopePhaseParams {
                dir:   adsr.release_dir(),
                mode:  adsr.release_mode(),
                shift: adsr.release_shift().as_u16() as _,
                step:  adsr.release_step().as_u16() as _,
            },
        }
    }

    pub fn phase_transitions(&mut self) {
        const N: usize = MAX_SIMD_WIDTH / 2;
        const CHUNKS: usize = 24 / N;
        let attack = Simd::<_, N>::splat(EnvelopePhase::Attack as u16);
        let attack_end_level = Simd::<i16, N>::splat(0x7fff);
        let decay = Simd::<_, N>::splat(EnvelopePhase::Decay as u16);
        let sustain = Simd::<_, N>::splat(EnvelopePhase::Sustain as u16);
        for ch in 0..CHUNKS {
            let base = ch * N;
            let levels = Simd::from_slice(&self.envelopes.level[base..]);
            let sustain_levels = Simd::from_slice(&self.envelopes.sustain_level[base..]);
            let phases = unsafe {
                std::ptr::read(self.envelopes.phase[base..].as_ptr() as *const Simd<u16, N>)
            };
            let is_attack = phases.simd_eq(attack);
            let is_attack_end_level = levels.simd_eq(attack_end_level);
            let attack_to_decay = is_attack & is_attack_end_level;
            let updated = attack_to_decay.select(decay, phases);

            let is_decay = updated.simd_eq(decay);
            let level_below_sustain = levels.simd_le(sustain_levels);
            let to_sustain = is_decay & level_below_sustain;
            let updated = to_sustain.select(sustain, updated);

            unsafe {
                std::ptr::copy_nonoverlapping(
                    updated.as_array() as *const u16,
                    self.envelopes.phase[base..].as_mut_ptr() as *mut u16,
                    N,
                );
            }
        }

        self.phase_transitions_scalar(CHUNKS * N..24);
    }

    #[inline(always)]
    fn phase_transitions_scalar(&mut self, range: impl Into<Range<usize>>) {
        let range = range.into();
        for i in range {
            if matches!(self.envelopes.phase[i], EnvelopePhase::Attack)
                && self.envelopes.level[i] == 0x7fff
            {
                self.envelopes.phase[i] = EnvelopePhase::Decay;
            }
            if matches!(self.envelopes.phase[i], EnvelopePhase::Decay)
                && self.envelopes.level[i] <= self.envelopes.sustain_level[i]
            {
                self.envelopes.phase[i] = EnvelopePhase::Sustain;
            }
        }
    }
}
