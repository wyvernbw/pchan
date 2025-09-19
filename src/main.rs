#![feature(test)]
#![feature(duration_millis_float)]
#![allow(unused_variables)]

use pchan_emu::Emu;
use std::{hint::black_box, time::Instant};

#[cfg(test)]
pub mod bench;

pub fn write_test_program(emu: &mut Emu) {
    use pchan_emu::cpu::ops::prelude::*;

    let main = program([
        addiu(8, 0, 0),           // ;  0 $t0 = 0
        addiu(10, 0, 4),          // ;  4 $t2 = 4
        addiu(9, 8, 0x0000_2000), // ;  8 calculate address $t1 = $t0 + 0x0000_2000
        sb(8, 9, 0),              // ; 12 store $i at $t1
        beq(8, 10, 16),           // ; 16 if $t0=$t2(4) jump by 16+8 to reach 40
        nop(),                    // ; 20
        addiu(8, 8, 1),           // ; 24 $t0 = $t0 + 1
        nop(),                    // ; 28
        j(-24),                   // ; 32 jump to 8 (return to beginning of loop)
        nop(),                    // ; 36
        nop(),                    // ; 40
        OpCode(69420),            // ; 44 halt
    ]);

    emu.mem.write_many::<u32>(emu.cpu.pc, &main);
}
pub fn time<F, T>(f: F) -> (T, f64)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let res = f();
    let end = start.elapsed();

    let runtime_ms = end.as_millis_f64();
    (res, runtime_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[allow(clippy::unit_arg)]
    black_box({
        let mut emu = Emu::default();
        write_test_program(&mut emu);
        _ = emu.step_jit();
    });
    Ok(())
}
