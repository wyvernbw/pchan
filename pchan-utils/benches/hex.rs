use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pchan_utils::hex;

pub fn hex_formatting_benchmark(c: &mut Criterion) {
    let number = 0xDEAD_BEEFu32;
    c.bench_function("pchan_utils::hex", |b| b.iter(|| hex(black_box(&number))));
    c.bench_function("std::fmt::format", |b| {
        b.iter(|| format!("0x{:X}", black_box(number)))
    });
}

criterion_group!(benches, hex_formatting_benchmark);
criterion_main!(benches);
