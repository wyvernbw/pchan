mod adpcm;
pub mod adsr;
mod gauss_interp;
pub mod reverb;

use core::alloc::Allocator;
use std::sync::Mutex;

use arbitrary_int::prelude::Integer;
use bitbybit::bitfield;
use pchan_bind::ringbuf::traits::Producer;
use pchan_bind::{AudioProducer, BindAudioProducer};
use pchan_utils::hex;

use crate::Emu;
use crate::io::evque::{EvCtx, EventId};
use crate::io::{CastIOFrom, CastIOInto, IOResult, UnhandledIO};
use crate::memory::kb;
use crate::spu::adpcm::{ADPCMCurrent, ADPCMHeader, ADPCMRepeat, ADPCMSampleRate, ADPCMStart};
use crate::spu::adsr::{ADSRState, EnvelopePhase, apply_volume};
use crate::spu::reverb::Reverb;

#[derive(derive_more::Debug)]
pub struct SpuState<A: Allocator> {
    voices:      Box<[Voice; 24], A>,
    adsr:        ADSRState,
    voice_flags: VoiceFlags,
    ctrl:        SpuCtrl,
    reverb:      Reverb<A>,
    #[debug(skip)]
    mem:         Box<[u16], A>,

    ram_start:   u16,
    /// internal register
    ram_current: usize,
    clock:       u64,

    prod:      Option<Mutex<AudioProducer>>,
    clock_idx: u64,
}

impl<A: Allocator + Copy> SpuState<A> {
    pub fn new(alloc: A) -> Self {
        let mut spu = Self {
            voices:      Box::new_in(Default::default(), alloc),
            voice_flags: VoiceFlags::default(),
            reverb:      Reverb::new(alloc),
            ctrl:        SpuCtrl::default(),
            mem:         create_spu_mem(alloc),
            ram_start:   0,
            ram_current: 0,
            clock:       0,
            prod:        None,
            adsr:        ADSRState::default(),
            clock_idx:   0,
        };
        for (i, voice) in spu.voices.iter_mut().enumerate() {
            voice.idx = i;
        }
        spu
    }
}

impl<A: Allocator + Clone> Clone for SpuState<A> {
    fn clone(&self) -> Self {
        Self {
            voices:      Box::clone(&self.voices),
            voice_flags: self.voice_flags.clone(),
            ctrl:        self.ctrl,
            mem:         self.mem.clone(),
            reverb:      self.reverb.clone(),
            ram_start:   self.ram_start,
            ram_current: self.ram_current,
            clock:       self.clock,
            prod:        None,
            adsr:        self.adsr.clone(),
            clock_idx:   0,
        }
    }
}

impl<A: Allocator> SpuState<A> {
    const MEM_SIZE: usize = kb(512);
    const CLOCK_CYCLES: u64 = 768;

    pub fn take_prod(&mut self) -> Option<Mutex<AudioProducer>> {
        self.prod.take()
    }
    pub fn put_prod(&mut self, prod: Option<Mutex<AudioProducer>>) {
        self.prod = prod;
    }
}

fn create_spu_mem<A: Allocator>(alloc: A) -> Box<[u16], A> {
    unsafe {
        Box::<[u16], A>::new_zeroed_slice_in(SpuState::<A>::MEM_SIZE / 2, alloc).assume_init()
    }
}

#[derive(Default, derive_more::Debug, Clone)]
struct Voice {
    start:       ADPCMStart,
    current:     ADPCMCurrent,
    repeat:      ADPCMRepeat,
    rate:        ADPCMSampleRate,
    idx:         usize,
    decode_buf:  [i16; 28],
    keyed_on:    bool,
    reached_end: bool,

    // adpcm decode sample history
    s1: i16,
    s2: i16,

    // gaussian interp sample history
    interp1: i16,
    interp2: i16,
    interp3: i16,

    /// interpolated sample
    current_sample: i16,
    pitch_counter:  u16,
    current_idx:    u8,
}

#[derive(Default, derive_more::Debug, Clone)]
struct VoiceFlags {
    key_on:  [VoiceOnOff; 2],
    key_off: [VoiceOnOff; 2],
    endx:    Endx,
}

#[bitfield(u16, debug)]
#[derive(Default)]
struct VoiceOnOff {
    #[bit(0, rw)]
    on: [bool; 16],
}

#[bitfield(u32, debug, default = 0xffffffff)]
struct Endx {
    #[bit(0, rw)]
    on: [bool; 24],
}

fn voice_idx(addr: u32, base: u32, stride: u32) -> Option<usize> {
    let addr = addr - base;
    if (addr).is_multiple_of(stride) {
        Some((addr / stride) as usize)
    } else {
        None
    }
}

impl<A: Allocator> Emu<A> {
    #[pchan_macros::instrument(level = "trace", skip(self), "spu:r")]
    pub fn spu_read<T: Copy>(&mut self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        // TODO add reads
        match address {
            // Sound RAM Data Transfer Address
            0x1f801da6 => Ok(u32::from(self.spu.ram_start).io_from_u32()),
            // adpcm sample rate
            addr @ 0x1f801c04..=0x1f801d7f if let Some(n) = voice_idx(addr, 0x1f801c04, 0x10) => {
                Ok(u32::from(self.spu.voices[n].rate.0).io_from_u32())
            }
            // adpcm start
            addr @ 0x1f801c06..=0x1f801d7f if let Some(n) = voice_idx(addr, 0x1f801c06, 0x10) => {
                Ok(u32::from(self.spu.voices[n].start.0).io_from_u32())
            }
            // adpcm repeat
            addr @ 0x1f801c0e..=0x1f801d7e if let Some(n) = voice_idx(addr, 0x1f801c0e, 0x10) => {
                Ok(u32::from(self.spu.voices[n].repeat.0).io_from_u32())
            }
            addr @ 0x1f801c00..=0x1f801d70 if let Some(n) = voice_idx(addr, 0x1f801c00, 0x10) => {
                Ok((self.spu.adsr.voice_left.registers[n]).io_from_u32())
            }
            addr @ 0x1f801c02..=0x1f801d72 if let Some(n) = voice_idx(addr, 0x1f801c02, 0x10) => {
                Ok((self.spu.adsr.voice_right.registers[n]).io_from_u32())
            }
            // ADSR volume
            // TODO: write
            addr @ 0x1f801c0c..=0x1f801d7c if let Some(n) = voice_idx(addr, 0x1f801c0c, 0x10) => {
                Ok((self.spu.adsr.envelopes.level[n]).io_from_u32())
            }
            // Voice 0..23 ON/OFF (status) (ENDX) (R)
            0x1f801d9c => {
                let endx = self.spu.voice_flags.endx.raw_value();
                tracing::info!(endx = %hex(endx));
                Ok(endx.io_from_u32())
            }
            // main volume left
            0x1f801d80 => Ok(self.spu.adsr.main_l.registers[0].io_from_u32()),
            // main volume right
            0x1f801d82 => Ok(self.spu.adsr.main_r.registers[0].io_from_u32()),

            0x1f801d84 => Ok(self.spu.reverb.io.lr[0].v_out.io_from_u32()),
            0x1f801d86 => Ok(self.spu.reverb.io.lr[1].v_out.io_from_u32()),
            0x1f801da2 => Ok(self.spu.reverb.esa.io_from_u32()),

            0x1f801d98 => Ok(self.spu.reverb.on_off.lower().io_from_u32()),
            0x1f801d9a => Ok(self.spu.reverb.on_off.upper().io_from_u32()),

            0x1f801daa => Ok(self.spu.ctrl.io_from_u32()),

            0x1f801db8 => Ok(self.spu.adsr.main_l.internal[0].io_from_u32()),
            0x1f801dba => Ok(self.spu.adsr.main_r.internal[0].io_from_u32()),

            0x1f801dc0..=0x1f801dfe => Ok(self.spu_reverb_read(address).io_from_u32()),

            addr @ 0x1f801e00..=0x1f801e5c if let Some(n) = voice_idx(addr, 0x1f801e00, 0x4) => {
                Ok(self.spu.adsr.voice_left.internal[n].io_from_u32())
            }
            addr @ 0x1f801e02..=0x1f801e62 if let Some(n) = voice_idx(addr, 0x1f801e02, 0x4) => {
                Ok(self.spu.adsr.voice_right.internal[n].io_from_u32())
            }

            _ => Err(crate::io::UnhandledIO(address)),
        }
    }
    #[pchan_macros::instrument(level = "trace", skip(self, value), "spu:w")]
    pub fn spu_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        if size_of::<T>() == 4 {
            let value = value.io_into_u32().to_le_bytes();
            let value = [
                u16::from_le_bytes([value[0], value[1]]),
                u16::from_le_bytes([value[2], value[3]]),
            ];
            self.spu_write(address, value[0])?;
            self.spu_write(address + 0x2, value[1])?;
            return Ok(());
        }

        let address = address & 0x1fffffff;
        let value = value.io_into_u32() as u16;
        match address {
            // Sound RAM Data Transfer Address
            0x1f801da6 => {
                self.spu.ram_start = value;
                self.spu.ram_current = (value as usize) << 2;
                Ok(())
            }
            // Sound RAM Data Transfer Fifo
            0x1f801da8 => {
                let current = self.spu.ram_current;
                self.spu.mem[current] = value;
                self.spu.ram_current += 1;
                Ok(())
            }
            // voices - adpcm sample rate
            addr @ 0x1f801c04..=0x1f801d7f if let Some(n) = voice_idx(addr, 0x1f801c04, 0x10) => {
                self.spu.voices[n].rate = ADPCMSampleRate(value);
                Ok(())
            }
            // voices - adpcm start
            addr @ 0x1f801c06..=0x1f801d76 if let Some(n) = voice_idx(addr, 0x1f801c06, 0x10) => {
                self.spu.voices[n].start = ADPCMStart(value);
                Ok(())
            }
            // ADSR volume
            addr @ 0x1f801c0c..=0x1f801d7c if let Some(n) = voice_idx(addr, 0x1f801c0c, 0x10) => {
                self.spu.adsr.envelopes.level[n] = value as i16;
                Ok(())
            }
            // voices - adpcm repeat
            addr @ 0x1f801c0e..=0x1f801d7e if let Some(n) = voice_idx(addr, 0x1f801c0e, 0x10) => {
                self.spu.voices[n].repeat = ADPCMRepeat(value);
                Ok(())
            }
            // voices - key on
            0x1f801d88 | 0x1f801d8a => {
                let key_idx = (address - 0x1f801d88) >> 1;
                let key_idx = key_idx as usize;

                self.spu.set_keys::<true>(key_idx, value);

                tracing::debug!(
                    "key_on.{} = {}",
                    key_idx,
                    hex(self.spu.voice_flags.key_on[key_idx])
                );
                Ok(())
            }
            // voices - key off
            0x1f801d8c | 0x1f801d8e => {
                let key_idx = (address - 0x1f801d8c) >> 1;
                let key_idx = key_idx as usize;

                self.spu.set_keys::<false>(key_idx, value);

                Ok(())
            }
            // adsr - voice volume left
            addr @ 0x1f801c00..=0x1f801d70 if let Some(n) = voice_idx(addr, 0x1f801c00, 0x10) => {
                self.spu.adsr.voice_left.set_register(n, value);
                Ok(())
            }
            // adsr - voice volume right
            addr @ 0x1f801c02..=0x1f801d72 if let Some(n) = voice_idx(addr, 0x1f801c02, 0x10) => {
                self.spu.adsr.voice_right.set_register(n, value);
                Ok(())
            }
            // adsr - envelope n lower bits
            addr @ 0x1f801c08..=0x1f801d78 if let Some(n) = voice_idx(addr, 0x1f801c08, 0x10) => {
                self.spu.adsr.set_register(n, 0, value);
                Ok(())
            }
            // adsr - envelope n upper bits
            addr @ 0x1f801c0a..=0x1f801d7a if let Some(n) = voice_idx(addr, 0x1f801c0a, 0x10) => {
                self.spu.adsr.set_register(n, 1, value);
                Ok(())
            }
            // main volume left
            0x1f801d80 => {
                self.spu.adsr.main_l.set_register(0, value);
                Ok(())
            }
            // main volume right
            0x1f801d82 => {
                self.spu.adsr.main_r.set_register(0, value);
                Ok(())
            }

            0x1f801d84 => {
                self.spu.reverb.io.lr[0].v_out = value as i16;
                Ok(())
            }
            0x1f801d86 => {
                self.spu.reverb.io.lr[1].v_out = value as i16;
                Ok(())
            }
            0x1f801da2 => {
                self.spu.reverb.esa = value;
                self.spu.reverb.bfa = value.as_u32() << 2;
                Ok(())
            }

            0x1f801db8 => {
                self.spu.adsr.main_l.internal[0] = value as i16;
                Ok(())
            }
            0x1f801dba => {
                self.spu.adsr.main_r.internal[0] = value as i16;
                Ok(())
            }

            0x1f801daa => {
                self.spu.ctrl = SpuCtrl::new_with_raw_value(value);
                Ok(())
            }

            0x1f801d98 => {
                self.spu.reverb.on_off.set_lower(value);
                Ok(())
            }
            0x1f801d9a => {
                self.spu.reverb.on_off.set_upper(value);
                Ok(())
            }

            0x1f801dc0..=0x1f801dfe => {
                self.spu_reverb_write(address, value);
                Ok(())
            }

            addr @ 0x1f801e00..=0x1f801e5c if let Some(n) = voice_idx(addr, 0x1f801e00, 0x4) => {
                self.spu.adsr.voice_left.internal[n] = value as _;
                Ok(())
            }
            addr @ 0x1f801e02..=0x1f801e62 if let Some(n) = voice_idx(addr, 0x1f801e02, 0x4) => {
                self.spu.adsr.voice_right.internal[n] = value as _;
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn spu_reverb_write(&mut self, address: u32, value: u16) {
        match address {
            0x1f801dc0 => self.spu.reverb.io.d_apf1 = value as _,
            0x1f801dc2 => self.spu.reverb.io.d_apf2 = value as _,
            0x1f801dc4 => self.spu.reverb.io.v_iir = value as _,
            0x1f801dc6 => self.spu.reverb.io.v_comb1 = value as _,
            0x1f801dc8 => self.spu.reverb.io.v_comb2 = value as _,
            0x1f801dca => self.spu.reverb.io.v_comb3 = value as _,
            0x1f801dcc => self.spu.reverb.io.v_comb4 = value as _,
            0x1f801dce => self.spu.reverb.io.v_wall = value as _,
            0x1f801dd0 => self.spu.reverb.io.v_apf1 = value as _,
            0x1f801dd2 => self.spu.reverb.io.v_apf2 = value as _,
            0x1f801dd4 => self.spu.reverb.io.lr[0].m_same = value as _,
            0x1f801dd6 => self.spu.reverb.io.lr[1].m_same = value as _,
            0x1f801dd8 => self.spu.reverb.io.lr[0].m_comb1 = value as _,
            0x1f801dda => self.spu.reverb.io.lr[1].m_comb1 = value as _,
            0x1f801ddc => self.spu.reverb.io.lr[0].m_comb2 = value as _,
            0x1f801dde => self.spu.reverb.io.lr[1].m_comb2 = value as _,
            0x1f801de0 => self.spu.reverb.io.lr[0].d_same = value as _,
            0x1f801de2 => self.spu.reverb.io.lr[1].d_same = value as _,
            0x1f801de4 => self.spu.reverb.io.lr[0].m_diff = value as _,
            0x1f801de6 => self.spu.reverb.io.lr[1].m_diff = value as _,
            0x1f801de8 => self.spu.reverb.io.lr[0].m_comb3 = value as _,
            0x1f801dea => self.spu.reverb.io.lr[1].m_comb3 = value as _,
            0x1f801dec => self.spu.reverb.io.lr[0].m_comb4 = value as _,
            0x1f801dee => self.spu.reverb.io.lr[1].m_comb4 = value as _,
            0x1f801df0 => self.spu.reverb.io.lr[0].d_diff = value as _,
            0x1f801df2 => self.spu.reverb.io.lr[1].d_diff = value as _,
            0x1f801df4 => self.spu.reverb.io.lr[0].m_apf1 = value as _,
            0x1f801df6 => self.spu.reverb.io.lr[1].m_apf1 = value as _,
            0x1f801df8 => self.spu.reverb.io.lr[0].m_apf2 = value as _,
            0x1f801dfa => self.spu.reverb.io.lr[1].m_apf2 = value as _,
            0x1f801dfc => self.spu.reverb.io.lr[0].v_in = value as _,
            0x1f801dfe => self.spu.reverb.io.lr[1].v_in = value as _,
            _ => unreachable!(),
        }
    }

    fn spu_reverb_read(&mut self, address: u32) -> u16 {
        match address {
            0x1f801dc0 => self.spu.reverb.io.d_apf1.io_from_u32(),
            0x1f801dc2 => self.spu.reverb.io.d_apf2.io_from_u32(),
            0x1f801dc4 => self.spu.reverb.io.v_iir.io_from_u32(),
            0x1f801dc6 => self.spu.reverb.io.v_comb1.io_from_u32(),
            0x1f801dc8 => self.spu.reverb.io.v_comb2.io_from_u32(),
            0x1f801dca => self.spu.reverb.io.v_comb3.io_from_u32(),
            0x1f801dcc => self.spu.reverb.io.v_comb4.io_from_u32(),
            0x1f801dce => self.spu.reverb.io.v_wall.io_from_u32(),
            0x1f801dd0 => self.spu.reverb.io.v_apf1.io_from_u32(),
            0x1f801dd2 => self.spu.reverb.io.v_apf2.io_from_u32(),
            0x1f801dd4 => self.spu.reverb.io.lr[0].m_same.io_from_u32(),
            0x1f801dd6 => self.spu.reverb.io.lr[1].m_same.io_from_u32(),
            0x1f801dd8 => self.spu.reverb.io.lr[0].m_comb1.io_from_u32(),
            0x1f801dda => self.spu.reverb.io.lr[1].m_comb1.io_from_u32(),
            0x1f801ddc => self.spu.reverb.io.lr[0].m_comb2.io_from_u32(),
            0x1f801dde => self.spu.reverb.io.lr[1].m_comb2.io_from_u32(),
            0x1f801de0 => self.spu.reverb.io.lr[0].d_same.io_from_u32(),
            0x1f801de2 => self.spu.reverb.io.lr[1].d_same.io_from_u32(),
            0x1f801de4 => self.spu.reverb.io.lr[0].m_diff.io_from_u32(),
            0x1f801de6 => self.spu.reverb.io.lr[1].m_diff.io_from_u32(),
            0x1f801de8 => self.spu.reverb.io.lr[0].m_comb3.io_from_u32(),
            0x1f801dea => self.spu.reverb.io.lr[1].m_comb3.io_from_u32(),
            0x1f801dec => self.spu.reverb.io.lr[0].m_comb4.io_from_u32(),
            0x1f801dee => self.spu.reverb.io.lr[1].m_comb4.io_from_u32(),
            0x1f801df0 => self.spu.reverb.io.lr[0].d_diff.io_from_u32(),
            0x1f801df2 => self.spu.reverb.io.lr[1].d_diff.io_from_u32(),
            0x1f801df4 => self.spu.reverb.io.lr[0].m_apf1.io_from_u32(),
            0x1f801df6 => self.spu.reverb.io.lr[1].m_apf1.io_from_u32(),
            0x1f801df8 => self.spu.reverb.io.lr[0].m_apf2.io_from_u32(),
            0x1f801dfa => self.spu.reverb.io.lr[1].m_apf2.io_from_u32(),
            0x1f801dfc => self.spu.reverb.io.lr[0].v_in.io_from_u32(),
            0x1f801dfe => self.spu.reverb.io.lr[1].v_in.io_from_u32(),
            _ => unreachable!(),
        }
    }

    pub fn handle_ev_spu_clock(&mut self, _ctx: EvCtx) {
        self.clock();
        let last_clock = self.spu.clock_idx * SpuState::<A>::CLOCK_CYCLES;
        self.evque.schedule_from(
            Self::handle_ev_spu_clock,
            EventId::default(),
            last_clock,
            SpuState::<A>::CLOCK_CYCLES,
        );
        self.spu.clock_idx += 1;
    }

    fn clock(&mut self) {
        self.spu.adsr.clock();
        let adsr = &mut self.spu.adsr;
        let flags = &mut self.spu.voice_flags;
        self.spu
            .voices
            .iter_mut()
            .enumerate()
            .for_each(|(idx, voice)| {
                voice.clock(&self.spu.mem, adsr);
                if voice.reached_end {
                    voice.reached_end = false;
                    flags.endx.set_on(idx, true);
                }
            });

        let mut mixed_l = 0i32;
        let mut mixed_r = 0i32;
        let mut reverb_l = 0i32;
        let mut reverb_r = 0i32;
        for i in 0..24 {
            let voice = &self.spu.voices[i];
            let adsr = &self.spu.adsr;
            let sample = voice.current_sample;

            let lvol =
                (i32::from(adsr.voice_left.internal[i]) * i32::from(adsr.envelopes.level[i])) >> 15;
            let rvol = (i32::from(adsr.voice_right.internal[i])
                * i32::from(adsr.envelopes.level[i]))
                >> 15;
            let lvol = lvol as i16;
            let rvol = rvol as i16;

            let left = apply_volume(sample, lvol);
            let right = apply_volume(sample, rvol);

            if self.spu.reverb.on_off.on_off(i) && self.spu.ctrl.reverb_on() {
                reverb_l += i32::from(left);
                reverb_r += i32::from(right);
            }

            mixed_l += i32::from(left);
            mixed_r += i32::from(right);
        }

        self.spu.reverb.downsample(
            reverb_l.saturating_truncate(),
            reverb_r.saturating_truncate(),
        );
        self.spu
            .reverb
            .apply_reverb(&mut self.spu.mem, self.spu.clock_idx as usize);
        self.spu
            .reverb
            .upsample_and_apply(&mut mixed_l, &mut mixed_r);

        let mixed_l = mixed_l.clamp(-0x8000, 0x7fff).truncate::<i16>();
        let mixed_r = mixed_r.clamp(-0x8000, 0x7fff).truncate::<i16>();

        let mixed_l = apply_volume(mixed_l, self.spu.adsr.main_l.internal[0]);
        let mixed_r = apply_volume(mixed_r, self.spu.adsr.main_r.internal[0]);

        if let Some(prod) = &mut self.spu.prod {
            _ = prod.get_mut().unwrap().prod.try_push(mixed_l);
            _ = prod.get_mut().unwrap().prod.try_push(mixed_r);
        }
    }
}

impl<A: Allocator> SpuState<A> {
    fn key_on(&mut self, idx: usize) {
        self.voices[idx].key_on(&self.mem, &mut self.adsr);
        self.adsr.envelopes.level[idx] = 0;
        self.adsr.envelopes.phase[idx] = EnvelopePhase::Attack;
    }

    fn key_off(&mut self, idx: usize) {
        self.adsr.envelopes.phase[idx] = EnvelopePhase::Release;
    }

    #[inline(always)]
    pub fn push(&mut self, value: u16) {
        let current = self.ram_current;
        self.mem[current] = value;
        self.ram_current += 1;
    }

    #[inline(always)]
    pub fn pop(&mut self) -> u16 {
        let value = self.mem[self.ram_current];
        self.ram_current += 1;
        value
    }
}

impl Voice {
    fn clock(&mut self, spu_ram: &[u16], adsr: &mut ADSRState) {
        let rate = self.rate.0.clamp(0x0, 0x4000);
        self.pitch_counter += rate;

        // 0x1000 = 44.1khz
        while self.pitch_counter >= 0x1000 {
            self.pitch_counter -= 0x1000;

            // Shift Gaussian history
            self.interp3 = self.interp2;
            self.interp2 = self.interp1;
            self.interp1 = self.decode_buf[self.current_idx as usize];

            self.current_idx += 1;

            if self.current_idx == 28 {
                self.current_idx = 0;
                self.advance_decode(spu_ram, adsr);
            }
        }

        self.current_sample = self.gauss_interpolation(self.decode_buf[self.current_idx as usize]);
    }

    fn gauss_interpolation(&self, current: i16) -> i16 {
        gauss_interp::interpolate(
            ((self.pitch_counter >> 4) & 0xFF) as usize,
            current,
            self.interp1,
            self.interp2,
            self.interp3,
        )
    }

    fn key_on(&mut self, spu_ram: &[u16], adsr: &mut ADSRState) {
        self.current = ADPCMCurrent(self.start.0);
        self.current_idx = 0;
        self.pitch_counter = 0x0;
        self.keyed_on = true;
        self.s2 = 0;
        self.s1 = 0;
        self.advance_decode(spu_ram, adsr);
    }

    fn key_off(&mut self) {
        self.keyed_on = false;
    }

    fn advance_decode(&mut self, spu_ram: &[u16], adsr: &mut ADSRState) {
        // address needs to be shifted right by 3 and we divide by 2 to get
        // offset in [u16] buffer, so the shift by 3 becomes a shift by 2.
        let address = u32::from(self.current.0) << 2;

        // a block is 16 bytes: 2 bytes header and 14 bytes samples, for 28
        // samples in total
        let block = &spu_ram[address as usize..];
        let block = &block[..8];
        let block: Result<&[u16; 8], _> = block.try_into();
        let Ok(block) = block else {
            return;
        };

        adpcm::decode_adpcm(block, &mut self.decode_buf, &mut self.s1, &mut self.s2);

        let header = ADPCMHeader::from_u16(block[0]);

        if header.flags.loop_start() {
            self.repeat = ADPCMRepeat(self.current.0);
        }

        // current holds the address shifted right by 3, so we add 2 to advance
        // by 16 bytes.
        self.current.0 = self.current.0.wrapping_add(2);
        if header.flags.loop_end() {
            self.reached_end = true;
            self.current = ADPCMCurrent(self.repeat.0);
            adsr.key_off(self.idx);
            adsr.envelopes.level[self.idx] = 0;
        }
    }
}

impl<A: Allocator> BindAudioProducer for Emu<A> {
    fn bind_producer(&mut self, prod: AudioProducer) {
        self.spu.prod = Some(prod.into());
    }
}

impl<A: Allocator> SpuState<A> {
    fn set_keys<const ON: bool>(&mut self, key_idx: usize, value: u16) {
        let keys = match ON {
            true => &mut self.voice_flags.key_on,
            false => &mut self.voice_flags.key_off,
        };

        let new_key = VoiceOnOff::new_with_raw_value(value);
        keys[key_idx] = new_key;

        let voice_offset = key_idx * 16;
        let len = if key_idx == 0 { 16 } else { 8 };

        let adsr = &mut self.adsr;
        let flags = &mut self.voice_flags;
        self.voices
            .iter_mut()
            .skip(voice_offset)
            .take(len)
            .enumerate()
            .filter(|(idx, _)| new_key.on(*idx))
            .for_each(|(idx, voice)| {
                let abs_idx = voice_offset + idx;
                if ON {
                    voice.key_on(&self.mem, adsr);
                    adsr.key_on(abs_idx);
                    flags.endx.set_on(abs_idx, false);
                } else {
                    voice.key_off();
                    adsr.key_off(abs_idx);
                }
            });
    }
}

/// # `SpuCtrl`
///
/// ```plaintext
///  15    SPU Enable                  (0=Off, 1=On)       (Don't care for CD Audio)
///  14    Mute SPU                    (0=Mute, 1=Unmute)  (Don't care for CD Audio)
///  13-10 Noise Frequency Shift       (0..0Fh = Low .. High Frequency)
///  9-8   Noise Frequency Step        (0..03h = Step "4,5,6,7")
///  7     Reverb Master Enable        (0=Disabled, 1=Enabled)
///  6     IRQ9 Enable                 (0=Disabled/Acknowledge, 1=Enabled; only when Bit15=1)
///  5-4   Sound RAM Transfer Mode     (0=Stop, 1=ManualWrite, 2=DMAwrite, 3=DMAread)
///  3     I2SB (PIO)    Reverb Enable (0=Off, 1=On)
///  2     I2SA (CD-ROM) Reverb Enable (0=Off, 1=On) (for CD-DA and XA-ADPCM)
///  1     I2SB (PIO)    Input Enable  (0=Off, 1=On)
///  0     I2SA (CD-ROM) Input Enable  (0=Off, 1=On) (for CD-DA and XA-ADPCM)
/// ```
#[bitfield(u16, debug)]
#[derive(Default)]
struct SpuCtrl {
    #[bit(7, rw)]
    reverb_on: bool,
}
