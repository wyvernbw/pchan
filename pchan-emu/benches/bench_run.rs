#![feature(test)]
use std::time::Instant;

use pchan_emu::{
    Emu,
    dynarec_v2::PipelineV2,
    io::vblank::{CPU_FREQ, NTSC_CYCLES},
};
use pchan_utils::setup_tracing;
#[cfg(test)]
use test::Bencher;

const TARGET: u32 = NTSC_CYCLES;
const RUNS: u32 = CPU_FREQ / TARGET;

extern crate test;

#[cfg(test)]
#[bench]
fn main(bench: &mut Bencher) {
    use pchan_emu::bootloader::Bootloader;

    setup_tracing();
    let mut emu = Emu::default();
    emu.load_bios().unwrap();
    emu.cpu.jump_to_bios();
    emu.cpu.vblank_timer = 1;

    let start = Instant::now();
    bench
        .bench(move |_| {
            use std::sync::atomic::Ordering;

            use pchan_emu::dynarec_v2::{
                BLOCKS_COMPILED, BLOCKS_EXECUTED, CACHE_HITS, CACHE_MISSES, INSTR_COMPILED,
                INSTR_EXECUTED,
            };

            let mut start_cycles;
            let mut dynarec = Box::default();
            for _ in 0..(RUNS * 32) {
                start_cycles = emu.cpu.cycles;
                loop {
                    use pchan_emu::dynarec_v2::{self};

                    dynarec = dynarec_v2::run_step(&mut emu, dynarec);
                    // Break when we've advanced by at least one vblank period
                    if emu.cpu.cycles >= start_cycles + NTSC_CYCLES as u64 {
                        break;
                    }
                }
            }

            let elapsed = start.elapsed();
            let frequency = emu.cpu.cycles as f64 / elapsed.as_secs_f64();
            let frequency = frequency / 1_000_000.0;
            tracing::info!(elapsed = elapsed.as_secs_f64());
            tracing::info!(emu.cpu.cycles);
            tracing::info!("frequency={}Mhz", frequency);

            // Report
            let blocks_exec = BLOCKS_EXECUTED.load(Ordering::Relaxed);
            let inst_exec = INSTR_EXECUTED.load(Ordering::Relaxed);
            tracing::info!("Blocks executed: {}", blocks_exec);
            tracing::info!("Instructions executed: {}", inst_exec);
            tracing::info!(
                "Average block size: {:.2}",
                inst_exec as f64 / blocks_exec as f64
            );
            tracing::info!(
                "IPS: {:.2}M",
                (inst_exec as f64 / elapsed.as_secs_f64()) / 1_000_000.0
            );
            let hits = CACHE_HITS.load(Ordering::Relaxed);
            let misses = CACHE_MISSES.load(Ordering::Relaxed);
            tracing::info!(
                "Cache hit rate: {:.2}%",
                (hits as f64 / (hits + misses) as f64) * 100.0
            );
            Ok(())
        })
        .unwrap();
}
