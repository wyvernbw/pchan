#![feature(iter_array_chunks)]
#![feature(portable_simd)]

use criterion::{Criterion, criterion_group, criterion_main};
use pchan_emu::{
    Emu,
    cpu::ops::{DecodedOp, NOP, OpCode},
    memory::{Memory, ext},
};
use rayon::{ThreadPool, ThreadPoolBuilder, iter::ParallelBridge};
use std::{hint::black_box, simd::Simd};

fn scalar_decode(n: usize, mem: &Memory) -> color_eyre::Result<()> {
    let mut sink = Vec::with_capacity(n);
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .take(black_box(n))
    {
        let op = mem.read::<u32, ext::NoExt>(address);
        let op = OpCode(op);
        let op = DecodedOp::try_from(op).unwrap_or(DecodedOp::NOP(NOP));
        sink.push(op);
        black_box(op);
    }
    Ok(())
}

fn vector_decode(n: usize, mem: &Memory) {
    const WIDTH: usize = pchan_emu::MAX_SIMD_WIDTH / size_of::<u32>();
    let mut sink = Vec::with_capacity(n);
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .step_by(black_box(WIDTH))
        .take(black_box(n / WIDTH))
    {
        let ops = mem.read::<Simd<u32, WIDTH>, ext::NoExt>(address);
        let ops = DecodedOp::extract_fields_simd(&ops);
        let ops = DecodedOp::decode(ops);
        sink.extend_from_slice(&ops);
        black_box(ops);
    }
}

fn scalar_new_decode(n: usize, mem: &Memory) {
    let mut sink = Vec::with_capacity(n);
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .take(black_box(n))
    {
        let op = mem.read::<u32, ext::NoExt>(address);
        let op = OpCode(op);
        let op = DecodedOp::extract_fields(&op);
        let op = DecodedOp::decode([op]);
        sink.push(op);
        black_box(op);
    }
}

fn scalar_new_parallel(n: usize, mem: &Memory, pool: &ThreadPool) {
    let (tx, rx) = crossbeam::channel::bounded(n);
    pool.scope_fifo(move |scope| {
        let tx = tx.clone();
        for address_chunk in (0xBFC0_0000..)
            .take(black_box(10))
            .cycle()
            .take(n)
            .array_chunks::<32>()
        {
            let tx = tx.clone();
            scope.spawn_fifo(move |_| {
                let chunk = address_chunk.map(|address| {
                    let op = mem.read::<u32, ext::NoExt>(address);
                    let op = OpCode(op);
                    let op = DecodedOp::extract_fields(&op);
                    let [op] = DecodedOp::decode([op]);
                    (address, op)
                });
                _ = tx.send(chunk);
                // sink.push(op);
            });
        }
    });
    let mut sink = rx.iter().collect::<Vec<_>>();
    sink.sort_by(|a, b| a[0].0.cmp(&b[0].0));
    black_box(sink);
}

fn scalar_new_lut(n: usize, mem: &Memory) {
    let mut sink = Vec::with_capacity(n);
    for address in (black_box(0xBFC0_0000..))
        .take(black_box(10))
        .cycle()
        .take(black_box(n))
    {
        let op = mem.read::<u32, ext::NoExt>(address);
        let op = OpCode(op);
        let op = DecodedOp::extract_fields(&op);
        let op = DecodedOp::lut_decode([op]);
        sink.push(op);
        black_box(op);
    }
}

fn decode_800(c: &mut Criterion) -> color_eyre::Result<()> {
    let mut emu = Emu::default();
    emu.load_bios()?;

    c.bench_function("scalar_decode_800 (chopped)", |b| {
        b.iter(|| scalar_decode(black_box(800), &emu.mem))
    });
    c.bench_function("vector_decode_800", |b| {
        b.iter(|| vector_decode(black_box(800), &emu.mem))
    });
    c.bench_function("scalar_new_decode_800", |b| {
        b.iter(|| scalar_new_decode(black_box(800), &emu.mem))
    });
    let pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    c.bench_function("scalar_new_parallel_800 (ass)", |b| {
        b.iter(|| {
            scalar_new_parallel(black_box(800), &emu.mem, &pool);
        })
    });
    c.bench_function("scalar_new_lut_800", |b| {
        b.iter(|| scalar_new_lut(800, &emu.mem))
    });
    Ok(())
}

criterion_group!(benches, decode_800);
criterion_main!(benches);
