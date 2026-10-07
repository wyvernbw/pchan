use core::alloc::Allocator;

use alloc::collections::VecDeque;
use arbitrary_int::prelude::*;
use bitbybit::bitfield;
use pchan_utils::MAX_SIMD_WIDTH;

use crate::spu::adsr::apply_volume_i32;

#[derive(Debug, Clone)]
pub struct Reverb<A: Allocator> {
    pub on_off: ReverbOnOff,

    downsamplers: [VecDeque<i32, A>; 2],
    upsamplers:   [VecDeque<i32, A>; 2],

    /// 1F801DA2h ESA      base    Reverb Work Area Start Address in Sound RAM
    pub esa: u16,
    pub bfa: u32,
    pub io:  ReverbIO,
}

/// # `ReverbIO`
///
/// ```plaintext
///  1F801DC0h dAPF1    disp    Reverb APF Offset 1
///  1F801DC2h dAPF2    disp    Reverb APF Offset 2
///  1F801DC4h vIIR     volume  Reverb Reflection Volume 1
///  1F801DC6h vCOMB1   volume  Reverb Comb Volume 1
///  1F801DC8h vCOMB2   volume  Reverb Comb Volume 2
///  1F801DCAh vCOMB3   volume  Reverb Comb Volume 3
///  1F801DCCh vCOMB4   volume  Reverb Comb Volume 4
///  1F801DCEh vWALL    volume  Reverb Reflection Volume 2
///  1F801DD0h vAPF1    volume  Reverb APF Volume 1
///  1F801DD2h vAPF2    volume  Reverb APF Volume 2
///  1F801DD4h mLSAME   src/dst Reverb Same Side Reflection Address 1 Left
///  1F801DD6h mRSAME   src/dst Reverb Same Side Reflection Address 1 Right
///  1F801DD8h mLCOMB1  src     Reverb Comb Address 1 Left
///  1F801DDAh mRCOMB1  src     Reverb Comb Address 1 Right
///  1F801DDCh mLCOMB2  src     Reverb Comb Address 2 Left
///  1F801DDEh mRCOMB2  src     Reverb Comb Address 2 Right
///  1F801DE0h dLSAME   src     Reverb Same Side Reflection Address 2 Left
///  1F801DE2h dRSAME   src     Reverb Same Side Reflection Address 2 Right
///  1F801DE4h mLDIFF   src/dst Reverb Different Side Reflect Address 1 Left
///  1F801DE6h mRDIFF   src/dst Reverb Different Side Reflect Address 1 Right
///  1F801DE8h mLCOMB3  src     Reverb Comb Address 3 Left
///  1F801DEAh mRCOMB3  src     Reverb Comb Address 3 Right
///  1F801DECh mLCOMB4  src     Reverb Comb Address 4 Left
///  1F801DEEh mRCOMB4  src     Reverb Comb Address 4 Right
///  1F801DF0h dLDIFF   src     Reverb Different Side Reflect Address 2 Left
///  1F801DF2h dRDIFF   src     Reverb Different Side Reflect Address 2 Right
///  1F801DF4h mLAPF1   src/dst Reverb APF Address 1 Left
///  1F801DF6h mRAPF1   src/dst Reverb APF Address 1 Right
///  1F801DF8h mLAPF2   src/dst Reverb APF Address 2 Left
///  1F801DFAh mRAPF2   src/dst Reverb APF Address 2 Right
///  1F801DFCh vLIN     volume  Reverb Input Volume Left
///  1F801DFEh vRIN     volume  Reverb Input Volume Right
/// ```
/// Other SPU registers that affect reverb:
/// ```plaintext
///  1F801D84h EVOLL    volume  Reverb Output Volume Left
///  1F801D86h EVOLR    volume  Reverb Output Volume Right
///  1F801DA2h ESA      base    Reverb Work Area Start Address in Sound RAM
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReverbIO {
    pub lr:      [BufLRConfig; 2],
    pub d_apf1:  u16,
    pub d_apf2:  u16,
    pub v_iir:   i16,
    pub v_comb1: i16,
    pub v_comb2: i16,
    pub v_comb3: i16,
    pub v_comb4: i16,
    pub v_wall:  i16,
    pub v_apf1:  i16,
    pub v_apf2:  i16,
}

#[derive(Debug, Clone, Default)]
pub struct BufLRConfig {
    pub m_same:  u16,
    pub m_comb1: u16,
    pub m_comb2: u16,
    pub d_same:  u16,
    pub m_diff:  u16,
    pub m_comb3: u16,
    pub m_comb4: u16,
    pub d_diff:  u16,
    pub m_apf1:  u16,
    pub m_apf2:  u16,
    pub v_in:    i16,
    pub v_out:   i16,
}

#[bitfield(u32, debug)]
#[derive(Default)]
#[must_use]
pub struct ReverbOnOff {
    /// io `0x1f801d98`
    #[bits(0..=15, rw)]
    lower: u16,
    /// io `1f801d9a`
    #[bits(16..=31, rw)]
    upper: u16,

    #[bit(0, r)]
    on_off: [bool; 32],
}

static FIR_FILTER: &[i32; 39] = &[
    -0x0001, 0x0000, 0x0002, 0x0000, -0x000A, 0x0000, 0x0023, 0x0000, -0x0067, 0x0000, 0x010A,
    0x0000, -0x0268, 0x0000, 0x0534, 0x0000, -0x0B90, 0x0000, 0x2806, 0x4000, 0x2806, 0x0000,
    -0x0B90, 0x0000, 0x0534, 0x0000, -0x0268, 0x0000, 0x010A, 0x0000, -0x0067, 0x0000, 0x0023,
    0x0000, -0x000A, 0x0000, 0x0002, 0x0000, -0x0001,
];

pub(crate) fn push_input_sample<A: Allocator>(buf: &mut VecDeque<i32, A>, sample: i16) {
    buf.pop_front();
    buf.push_back(i32::from(sample));
    debug_assert_eq!(buf.len(), FIR_FILTER.len());
}

pub fn apply_fir_filter<A: Allocator>(buf: &mut VecDeque<i32, A>) -> i32 {
    assert_eq!(buf.len(), FIR_FILTER.len());
    let buf = buf.make_contiguous();
    buf.iter()
        .zip(FIR_FILTER)
        .map(|(&s, &f)| s * f)
        .sum::<i32>()
        >> 15
}

pub fn apply_fir_filter_simd<A: Allocator>(buf: &mut VecDeque<i32, A>) -> i16 {
    use core::simd::prelude::*;

    const N: usize = MAX_SIMD_WIDTH / size_of::<i32>();
    const N2: usize = N * 2;

    assert_eq!(buf.len(), FIR_FILTER.len());
    let buf = buf.make_contiguous();

    let mut acc: [Simd<i32, N>; 2] = [Simd::splat(0); _];
    let (samples, samples_rem) = buf.as_chunks::<N2>();
    let (fir, fir_rem) = FIR_FILTER.as_chunks::<N2>();

    for (s, fir) in samples.iter().zip(fir) {
        let s1 = Simd::<_, N>::from_slice(&s[..N]);
        let s2 = Simd::<_, N>::from_slice(&s[N..]);
        let fir1 = Simd::<_, N>::from_slice(&fir[..N]);
        let fir2 = Simd::<_, N>::from_slice(&fir[N..]);
        acc[0] += s1 * fir1;
        acc[1] += s2 * fir2;
    }
    let mut sum = acc[0].reduce_sum() + acc[1].reduce_sum();

    for (s, fir) in samples_rem.iter().zip(fir_rem) {
        sum += (*s) * (*fir);
    }

    (sum >> 15).saturating_truncate()
}

impl<A: Allocator + Copy> Reverb<A> {
    pub fn new(alloc: A) -> Self {
        let mut r = Self {
            on_off:       ReverbOnOff::default(),
            downsamplers: core::array::from_fn(|_| VecDeque::with_capacity_in(39, alloc)),
            upsamplers:   core::array::from_fn(|_| VecDeque::with_capacity_in(39, alloc)),
            esa:          0,
            io:           ReverbIO::default(),
            bfa:          0,
        };
        r.downsamplers[0].resize(39, 0);
        r.downsamplers[1].resize(39, 0);
        r.upsamplers[0].resize(39, 0);
        r.upsamplers[1].resize(39, 0);
        r
    }
}

impl<A: Allocator> Reverb<A> {
    pub fn downsample(&mut self, sl: i16, sr: i16) {
        push_input_sample(&mut self.downsamplers[0], sl);
        push_input_sample(&mut self.downsamplers[1], sr);
    }

    fn get_sample(&mut self, idx: usize) -> i32 {
        apply_fir_filter(&mut self.downsamplers[idx % self.downsamplers.len()])
    }

    fn ring_addr(&self, addr: u16) -> usize {
        self.ring_addr_global(addr.as_u32().unbounded_shl(2))
            .as_usize()
    }
    fn ring_addr_global(&self, addr: u32) -> u32 {
        let base = self.esa.as_u32() << 2;
        let mut a = self.bfa + (addr & 0x3ffff);
        if a & 0x40000 != 0 {
            a += base;
        }
        a & 0x3ffff
    }

    pub fn apply_reverb(&mut self, ram: &mut [u16], lr: usize) {
        let lr = lr % 2;
        let other = (lr + 1) % 2;
        let sample = self.get_sample(lr);
        let sample = apply_volume_i32(sample, self.io.lr[lr].v_in.as_i32());

        // same side reflection
        self.reflection(ReflectionParams {
            ram,
            sample,
            m_addr: self.io.lr[lr].m_same,
            d_addr: self.io.lr[lr].d_same,
            v_wall: self.io.v_wall,
            v_iir: self.io.v_iir,
        });

        // different side reflection
        self.reflection(ReflectionParams {
            ram,
            sample,
            m_addr: self.io.lr[lr].m_diff,
            d_addr: self.io.lr[other].d_diff,
            v_wall: self.io.v_wall,
            v_iir: self.io.v_iir,
        });

        let m_comb1 = self.ring_addr(self.io.lr[lr].m_comb1);
        let m_comb2 = self.ring_addr(self.io.lr[lr].m_comb2);
        let m_comb3 = self.ring_addr(self.io.lr[lr].m_comb3);
        let m_comb4 = self.ring_addr(self.io.lr[lr].m_comb4);
        let comb_out = spu_mul(self.io.v_comb1, ram[m_comb1].as_i16())
            .as_i32()
            .saturating_add(spu_mul(self.io.v_comb2, ram[m_comb2].as_i16()).as_i32())
            .saturating_add(spu_mul(self.io.v_comb3, ram[m_comb3].as_i16()).as_i32())
            .saturating_add(spu_mul(self.io.v_comb4, ram[m_comb4].as_i16()).as_i32());

        let apf1 = self.apf(ApfParams {
            ram,
            input: comb_out,
            v_apf: self.io.v_apf1,
            m_apf: self.io.lr[lr].m_apf1,
            d_apf: self.io.d_apf1,
        });
        let apf2 = self.apf(ApfParams {
            ram,
            input: apf1,
            v_apf: self.io.v_apf2,
            m_apf: self.io.lr[lr].m_apf2,
            d_apf: self.io.d_apf2,
        });

        push_input_sample(&mut self.upsamplers[lr], apf2.saturating_truncate());
        push_input_sample(&mut self.upsamplers[other], 0);

        if lr == 1 {
            self.bfa = self.ring_addr_global(0x1)
        }
    }

    pub fn upsample_and_apply(&mut self, left: &mut i32, right: &mut i32) {
        let l = apply_fir_filter(&mut self.upsamplers[0]) << 1;
        let l = apply_volume_i32(l, self.io.lr[0].v_out.as_i32());
        let r = apply_fir_filter(&mut self.upsamplers[1]) << 1;
        let r = apply_volume_i32(r, self.io.lr[1].v_out.as_i32());

        *left += l;
        *right += r;
    }

    fn reflection(
        &self,
        ReflectionParams {
            ram,
            sample,
            m_addr,
            d_addr,
            v_wall,
            v_iir,
        }: ReflectionParams,
    ) {
        let m_units = m_addr.as_u32() << 2;
        let last_addr = self.ring_addr_global(m_units.wrapping_sub(1)) as usize;
        let m_addr = self.ring_addr(m_addr);
        let d_addr = self.ring_addr(d_addr);
        let last = ram[last_addr].as_i16().as_i32();

        let refl = spu_mul(ram[d_addr].as_i16(), v_wall).as_i32();
        let out = spu_mul(
            sample.as_i32().saturating_add(refl).saturating_sub(last),
            v_iir,
        )
        .as_i32()
        .saturating_add(last);
        ram[m_addr] = out.saturating_truncate::<i16>() as u16;
    }

    fn apf(
        &self,
        ApfParams {
            ram,
            input,
            v_apf,
            m_apf,
            d_apf,
        }: ApfParams,
    ) -> i32 {
        let ghost = self.ring_addr(m_apf.wrapping_sub(d_apf));
        let m_apf = self.ring_addr(m_apf);
        let buffered = input.saturating_sub(spu_mul(v_apf, ram[ghost].as_i16()).as_i32());
        ram[m_apf] = buffered.saturating_truncate::<i16>() as _;

        spu_mul(v_apf, buffered)
            .as_i32()
            .saturating_add(ram[ghost].as_i16().as_i32())
    }
}

struct ReflectionParams<'a> {
    ram:    &'a mut [u16],
    sample: i32,
    m_addr: u16,
    d_addr: u16,
    v_wall: i16,
    v_iir:  i16,
}

struct ApfParams<'a> {
    ram:   &'a mut [u16],
    input: i32,
    v_apf: i16,
    m_apf: u16,
    d_apf: u16,
}

fn spu_mul(a: impl Integer, b: impl Integer) -> i16 {
    ((a.as_i32().saturating_mul(b.as_i32())) >> 15).saturating_truncate()
}
