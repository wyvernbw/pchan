use std::mem::transmute;

use arbitrary_int::prelude::*;
use bitbybit::bitfield;

use derive_more as d;

/// mapped to `1F801C06h+N*10h`
#[derive(Default, derive_more::Debug, Clone, Copy, d::Deref, d::DerefMut)]
pub struct ADPCMStart(pub u16);
#[derive(Default, derive_more::Debug, Clone, Copy, d::Deref, d::DerefMut)]
pub struct ADPCMCurrent(pub u16);
/// mapped to `1F801C0Eh+N*10h`
#[derive(Default, derive_more::Debug, Clone, Copy, d::Deref, d::DerefMut)]
pub struct ADPCMRepeat(pub u16);
/// mapped to `1F801C04h+N*10h`
#[derive(Default, derive_more::Debug, Clone, Copy, d::Deref, d::DerefMut)]
pub struct ADPCMSampleRate(pub u16);

#[bitfield(u8)]
pub struct ADPCMShiftFilter {
    #[bits(0..=3, rw)]
    shift:  u4,
    #[bits(4..=6, rw)]
    filter: u3,
}

#[bitfield(u8)]
pub struct ADPCMFlags {
    #[bit(0, rw)]
    loop_end:    bool,
    #[bit(1, rw)]
    loop_repeat: bool,
    #[bit(2, rw)]
    loop_start:  bool,
}

#[repr(C)]
pub struct ADPCMHeader {
    pub shift_filter: ADPCMShiftFilter,
    pub flags:        ADPCMFlags,
}

impl ADPCMHeader {
    pub fn from_u16(value: u16) -> Self {
        unsafe { transmute(value) }
    }
}

/// ```md
///  00h       Shift/Filter (reportedly same as for CD-XA) (see there)
///  01h       Flag Bits (see below)
///  02h       Compressed Data (LSBs=1st Sample, MSBs=2nd Sample)
///  03h       Compressed Data (LSBs=3rd Sample, MSBs=4th Sample)
///  04h       Compressed Data (LSBs=5th Sample, MSBs=6th Sample)
///  ...       ...
///  0Fh       Compressed Data (LSBs=27th Sample, MSBs=28th Sample)
/// ```
pub fn decode_adpcm(from: &[u16; 8], to: &mut [i16], s1: &mut i16, s2: &mut i16, s3: &mut i16) {
    const POS_ADPCM_TABLE: [i32; 5] = [0, 60, 115, 98, 122];
    const NEG_ADPCM_TABLE: [i32; 5] = [0, 0, -52, -55, -60];

    let header = from[0];
    let header = ADPCMHeader::from_u16(header.to_le());
    let samples = &from[1..];
    let shift = header.shift_filter.shift().as_u8();
    let shift = if shift > 12 { 9 } else { shift };
    let shift = 12 - shift;

    let filter = header.shift_filter.filter().as_u8().min(4);

    // SAFETY: filter is between 0x0 and 0x4
    let (f0, f1) = unsafe {
        let f0 = POS_ADPCM_TABLE.get_unchecked(filter as usize);
        let f1 = NEG_ADPCM_TABLE.get_unchecked(filter as usize);
        (f0, f1)
    };

    // SAFETY: holy unsafe
    // this should be safe because samples is guaranteed at compile time
    // to have 7 u16 elements, so 14 bytes (u8). casting it to this type
    // and not a plain slice allows the compiler to remove bounds checks
    let samples = unsafe {
        (std::ptr::slice_from_raw_parts(
            samples.as_ptr() as *const u8,
            std::mem::size_of_val(samples),
        ) as *const [u8; 14])
            .as_ref_unchecked()
    };

    for (idx, dest) in to.iter_mut().take(28).enumerate() {
        let sample = samples[idx / 2];
        let sample = (sample >> (4 * (idx % 2))) & 0x0F;
        let sample = ((sample as i8) << 4 >> 4) as i32; // sign extend from bit 3

        {
            let s1 = *s1 as i32;
            let s2 = *s2 as i32;

            let sample = (sample << shift) + ((s1 * f0 + s2 * f1 + 32) / 64);
            let sample = sample.clamp(-0x8000, 0x7fff);
            *dest = sample as i16;
        }

        *s3 = *s2;
        *s2 = *s1;
        *s1 = *dest;
    }
}

#[cfg(test)]
#[test]
fn test_adpcm_decode() {
    use assert_hex::assert_eq_hex;

    pchan_utils::setup_tracing();
    let values: [u8; 16] = [
        0x48, 0x00, 0xd2, 0x4d, 0xef, 0xf0, 0xe3, 0x3c, 0x1f, 0xed, 0xf4, 0x2f, 0x2e, 0xef, 0xe3,
        0x13,
    ];

    let values = unsafe {
        (std::ptr::slice_from_raw_parts(values.as_ptr(), std::mem::size_of_val(&values))
            as *const [u16; 8])
            .as_ref_unchecked()
    };

    let mut decode_buf = [0i16; 28];
    let mut s1 = 392;
    let mut s2 = 465;
    decode_adpcm(values, &mut decode_buf, &mut s1, &mut s2, &mut 0);
    assert_eq_hex!(
        decode_buf,
        [
            343, 238, 84, 2, -90, -204, -304, -403, -434, -481, -573, -592, -606, -583, -590, -609,
            -543, -479, -419, -317, -242, -131, -38, 18, 118, 176, 273, 371
        ]
    );
}
