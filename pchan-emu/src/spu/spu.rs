pub mod adpcm;

use std::array::from_fn;
use std::sync::{Arc, Mutex};

use bitbybit::bitfield;
use pchan_bind::ringbuf::traits::*;
use pchan_bind::{AudioProducer, BindAudioProducer};
use pchan_utils::{CacheAligned, hex};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::Emu;
use crate::io::{CastIOInto, IO, IOResult, UnhandledIO};
use crate::memory::kb;
use crate::spu::adpcm::{ADPCMCurrent, ADPCMHeader, ADPCMRepeat, ADPCMSampleRate, ADPCMStart};

#[derive(derive_more::Debug)]
pub struct SpuState {
    voices:      Box<[CacheAligned<Voice>; 24]>,
    voice_flags: VoiceFlags,
    mem:         Box<[u16]>,

    ram_start:   u16,
    /// internal register
    ram_current: usize,
    clock:       u64,

    prod: Option<Mutex<AudioProducer>>,
}

impl Default for SpuState {
    fn default() -> Self {
        Self {
            voices:      Default::default(),
            voice_flags: Default::default(),
            mem:         create_spu_mem(),
            ram_start:   Default::default(),
            ram_current: Default::default(),
            clock:       Default::default(),
            prod:        Default::default(),
        }
    }
}

impl Clone for SpuState {
    fn clone(&self) -> Self {
        Self {
            voices:      self.voices.clone(),
            voice_flags: self.voice_flags.clone(),
            mem:         self.mem.clone(),
            ram_start:   self.ram_start,
            ram_current: self.ram_current,
            clock:       self.clock,
            prod:        None,
        }
    }
}

impl SpuState {
    const MEM_SIZE: usize = kb(512);
    const CLOCK_CYCLES: u64 = 768;
}

fn create_spu_mem() -> Box<[u16]> {
    vec![0u16; SpuState::MEM_SIZE / 2].into_boxed_slice()
}

#[derive(Default, derive_more::Debug, Clone)]
struct Voice {
    start:      ADPCMStart,
    current:    ADPCMCurrent,
    repeat:     ADPCMRepeat,
    rate:       ADPCMSampleRate,
    decode_buf: [i16; 28],

    /// old sample. used in adpcm decoding
    s1: i16,
    /// older sample. used in adpcm decoding
    s2: i16,

    pitch_counter:  u16,
    current_idx:    u8,
    current_sample: i16,
}

#[derive(Default, derive_more::Debug, Clone)]
struct VoiceFlags {
    key_on: [KeyOn; 2],
}

#[bitfield(u16, debug)]
#[derive(Default)]
struct KeyOn {
    #[bit(0, rw)]
    on: [bool; 16],
}

pub trait Spu: IO {
    #[pchan_macros::instrument(level = "trace", skip(self), "spu:r")]
    fn read<T: Copy>(&mut self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        // TODO add reads
        Err(crate::io::UnhandledIO(address))
    }
    #[pchan_macros::instrument(level = "trace", skip(self, value), "spu:w")]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        fn voice_idx(addr: u32, base: u32, stride: u32) -> Option<usize> {
            let addr = addr - base;
            if (addr).is_multiple_of(stride) {
                Some((addr / stride) as usize)
            } else {
                None
            }
        }

        let address = address & 0x1fffffff;
        let value = value.io_into_u32() as u16;
        match address {
            // Sound RAM Data Transfer Address
            0x1f801da6 => {
                self.spu_mut().ram_start = value;
                self.spu_mut().ram_current = (value as usize) << 2;
                Ok(())
            }
            // Sound RAM Data Transfer Fifo
            0x1f801da8 => {
                let spu = self.spu_mut();
                let current = spu.ram_current;
                spu.mem[current] = value;
                spu.ram_current += 1;
                Ok(())
            }
            // voices - adpcm sample rate
            addr @ 0x1f801c04..=0x1f801d7f if let Some(n) = voice_idx(addr, 0x1f801c04, 0x10) => {
                let spu = self.spu_mut();
                spu.voices[n].rate = ADPCMSampleRate(value);
                Ok(())
            }
            // voices - adpcm start
            addr @ 0x1f801c06..=0x1f801d7f if let Some(n) = voice_idx(addr, 0x1f801c06, 0x10) => {
                let spu = self.spu_mut();
                spu.voices[n].start = ADPCMStart(value);
                Ok(())
            }
            // voices - adpcm repeat
            addr @ 0x1f801c0e..=0x1f801d7f if let Some(n) = voice_idx(addr, 0x1f801c0e, 0x10) => {
                let spu = self.spu_mut();
                spu.voices[n].repeat = ADPCMRepeat(value);
                Ok(())
            }
            // voices - key on
            0x1f801d88 | 0x1f801d8a => {
                let key_idx = (address - 0x1f801d88) >> 1;
                let key_idx = key_idx as usize;
                let spu = self.spu_mut();
                spu.voice_flags.key_on[key_idx] = KeyOn::new_with_raw_value(value);

                let voice_offset = key_idx * 16;
                let len = if key_idx == 0 { 16 } else { 8 };

                tracing::info!(
                    "key_on.{} = {}",
                    key_idx,
                    hex(spu.voice_flags.key_on[key_idx])
                );
                // TODO benchmark vs sequential iterator.
                spu.voices
                    .iter_mut()
                    .skip(voice_offset)
                    .take(len)
                    .enumerate()
                    .filter(|(idx, _)| spu.voice_flags.key_on[key_idx].on(*idx))
                    .for_each(|(idx, voice)| {
                        tracing::info!("keyed on: {}", idx + voice_offset);
                        voice.key_on(&spu.mem);
                    });
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn run_spu(&mut self, mut dclock: u64) {
        dclock += self.spu().clock;
        {
            let count: u32 = self
                .spu()
                .voice_flags
                .key_on
                .map(|k| k.raw_value().count_ones())
                .iter()
                .sum();
            if count == 0 {
                return;
            }
        }
        self.spu_mut().clock = 0;
        while dclock >= SpuState::CLOCK_CYCLES {
            dclock -= SpuState::CLOCK_CYCLES;
            self.clock();
        }
        self.spu_mut().clock += dclock;
    }

    fn clock(&mut self) {
        let spu = self.spu_mut();

        // TODO benchmark vs sequential iterator.
        spu.voices
            .iter_mut()
            .enumerate()
            .filter(|(idx, _)| {
                let key_idx = idx / 16;
                spu.voice_flags.key_on[key_idx].on(*idx % 16)
            })
            .for_each(|(idx, voice)| {
                voice.clock(&spu.mem);
            });

        let mixed: i32 = spu
            .voices
            .iter()
            // TODO implement volume
            .map(|voice| voice.current_sample as i32 / 4)
            .sum();

        if let Some(prod) = &mut spu.prod {
            _ = prod.get_mut().unwrap().prod.try_push(mixed as i16);
        }
    }
}

impl Spu for Emu {}

impl Voice {
    fn clock(&mut self, spu_ram: &[u16]) {
        let rate = self.rate.0.clamp(0x0, 0x4000);
        self.pitch_counter += rate;

        // 0x1000 = 44.1khz
        while self.pitch_counter >= 0x1000 {
            self.pitch_counter -= 0x1000;
            self.current_idx += 1;
            if self.current_idx == 28 {
                self.current_idx = 0;
                tracing::info!("|> current {}", hex(self.current));
                self.advance_decode(spu_ram);
                tracing::info!("|> current {}", hex(self.current));
            }
        }

        self.current_sample = self.decode_buf[self.current_idx as usize];
    }

    fn key_on(&mut self, spu_ram: &[u16]) {
        self.current = ADPCMCurrent(self.start.0);
        self.current_idx = 0;
        self.pitch_counter = 0x0;
        // self.advance_decode(spu_ram);
    }

    fn advance_decode(&mut self, spu_ram: &[u16]) {
        // address needs to be shifted right by 3 and we divide by 2 to get
        // offset in [u16] buffer.
        let address = (self.current.0 as u32) << 2;
        tracing::info!("decode at {}", hex(address));

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

        if header.flags.loop_end() {
            self.current = ADPCMCurrent(self.repeat.0);
            if !header.flags.loop_repeat() {
                // TODO: mute the voice
            }
        } else {
            // current holds the address shifted right by 3, so we add 2 to advance
            // by 16 bytes.
            self.current.0 += 2;
        }
    }
}

impl BindAudioProducer for Emu {
    fn bind_producer(&mut self, prod: AudioProducer) {
        self.spu.prod = Some(prod.into());
    }
}
