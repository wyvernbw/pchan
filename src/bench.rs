use std::hint::black_box;

use pchan_emu::Emu;
use pchan_utils::setup_tracing;

use crate::{time, write_test_program};

#[test]
fn bench_single() {
    #[allow(clippy::unit_arg)]
    setup_tracing();
    let _: () = black_box({
        let mut emu = Emu::default();
        write_test_program(&mut emu);
        _ = emu.step_jit()
    });
}
#[test]
fn bench_jit() {
    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0;
    write_test_program(&mut emu);

    let mut average = 0.0;
    for _ in 0..100 {
        emu.mem.write_many::<u32>(0x0000_2000, &[0u32, 0u32]);
        emu.jit.clear_cache();
        emu.cpu.pc = 0;
        emu.cpu.clear_registers();
        let (_, elapsed) = time(|| {
            _ = black_box(emu.step_jit());
        });
        average += elapsed;
        let slice = &emu.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
        assert_eq!(slice, &[0, 1, 2, 3]);
    }
    average /= 100.0;
    let slice = &emu.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
    assert_eq!(slice, &[0, 1, 2, 3]);
    // println!("{:#?}", &emu);
    tracing::info!("{:?}", slice);
    tracing::info!("cache usage: {:?}", emu.jit.cache_usage());
    tracing::info!("no cache: took {average}ms");

    let mut average = 0.0;
    emu.cpu.pc = 0;
    _ = black_box(emu.step_jit());
    emu.cpu.pc = 0;
    for _ in 0..100 {
        emu.mem.write_many::<u32>(0x0000_2000, &[0u32, 0u32]);
        emu.cpu.pc = 0;
        emu.cpu.clear_registers();
        let (_, elapsed) = time(|| {
            _ = black_box(emu.step_jit());
        });
        average += elapsed;
        let slice = &emu.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
        assert_eq!(slice, &[0, 1, 2, 3]);
    }
    average /= 100.;
    let slice = &emu.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
    assert_eq!(slice, &[0, 1, 2, 3]);
    // println!("{:#?}", &emu);
    tracing::info!("{:?}", slice);
    tracing::info!("cache usage: {:?}", emu.jit.cache_usage());
    tracing::info!("w/ cache: took {average}ms");
}
