#![feature(portable_simd)]

use criterion::{Criterion, criterion_group, criterion_main};
use pchan_emu::{
    Emu,
    cpu::ops::{DecodedOp, OpCode},
    memory::{Memory, ext},
};
use std::{hint::black_box, simd::Simd};

fn scalar_decode(n: usize, mem: &Memory) -> color_eyre::Result<()> {
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .take(black_box(n))
    {
        let op = mem.read::<u32, ext::NoExt>(address);
        let op = OpCode(op);
        let op = DecodedOp::try_from(op);
        black_box(op);
    }
    Ok(())
}

fn vector_decode(n: usize, mem: &Memory) {
    const WIDTH: usize = pchan_emu::MAX_SIMD_WIDTH / size_of::<u32>();
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .step_by(black_box(WIDTH))
        .take(black_box(n / WIDTH))
    {
        let ops = mem.read::<Simd<u32, WIDTH>, ext::NoExt>(address);
        let ops = DecodedOp::extract_fields_simd(&ops);
        let ops = DecodedOp::decode(ops);
        black_box(ops);
    }
}

fn scalar_new_decode(n: usize, mem: &Memory) {
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .take(black_box(n))
    {
        let op = mem.read::<u32, ext::NoExt>(address);
        let op = OpCode(op);
        let op = DecodedOp::extract_fields(&op);
        let op = DecodedOp::decode([op]);
        black_box(op);
    }
}

fn decode_800(c: &mut Criterion) -> color_eyre::Result<()> {
    let mut emu = Emu::default();
    emu.load_bios()?;

    c.bench_function("scalar_decode_800", |b| {
        b.iter(|| black_box(scalar_decode(black_box(800), &emu.mem)))
    });
    c.bench_function("vector_decode_800", |b| {
        b.iter(|| black_box(vector_decode(black_box(800), &emu.mem)))
    });
    c.bench_function("scalar_new_decode_800", |b| {
        b.iter(|| black_box(scalar_new_decode(black_box(800), &emu.mem)))
    });
    Ok(())
}

criterion_group!(benches, decode_800);
criterion_main!(benches);
