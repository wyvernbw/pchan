use super::*;
use crate::memory::{MemRead, PhysAddr};
use pchan_utils::setup_tracing;
use pretty_assertions::{assert_eq, assert_matches};
use rstest::*;

#[rstest]
#[instrument]
fn test_pipeline_single_load(setup_tracing: ()) {
    tracing::info!("testing single load");
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 0x1000; // r9 = 0x1000
    mem.write(PhysAddr::new(0x0), Op::lw(8, 9, 4)); // Instruction at PC=0
    mem.write::<u32>(PhysAddr::new(0x1000 + 4), 0x12345678); // Data at 0x1004

    // Run 5 cycles to complete one load instruction
    for _ in 0..5 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    tracing::info!("{:08x}", cpu.reg[8]);

    // Verify $t0 (reg 8) contains the loaded value
    assert_eq!(cpu.reg[8], 0x12345678, "LW should load 0x12345678 into $t0");
}
#[rstest]
#[instrument]
fn test_pipeline_lb(setup_tracing: ()) {
    tracing::info!("testing LB instruction");
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[8] = 0x1000;
    mem.write(PhysAddr::new(0x0), Op::lbu(8, 8, 2));
    mem.write::<u8>(PhysAddr::new(0x1002), 0xAB); // byte to load

    for _ in 0..5 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(cpu.reg[8], 0xAB, "LB should load 0xAB into r8");
}

#[rstest]
#[instrument]
fn test_pipeline_multiple_loads(setup_tracing: ()) {
    tracing::info!("testing multiple loads in a row");
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 0x1000;
    cpu.reg[10] = 0x2000;

    let program = Program::new([
        Op::lw(8, 9, 4),
        Op::NOP,
        Op::lb(9, 8, 1),
        Op::lbu(10, 10, 2),
    ]);
    tracing::info!(%program);
    mem.write_all(PhysAddr::new(0), program);

    mem.write::<u32>(PhysAddr::new(0x1004), 0x1234); // LW source
    mem.write::<u8>(PhysAddr::new(0x1235), 0x7F); // LB source (0x1235 = r8 + 1)
    mem.write::<u8>(PhysAddr::new(0x2002), 0xFF); // LBU source (r10 + 2)

    // Run enough cycles to complete all three loads
    for _ in 0..15 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(cpu.reg[8], 0x1234, "LW should load 0x1234 into r8");
    assert_eq!(cpu.reg[9], 0x7F, "LB should load 0x7F into r9");
    assert_eq!(cpu.reg[10], 0xFF, "LBU should load 0xFF into r10");
}

#[rstest]
#[instrument]
fn test_single_add(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 42;
    cpu.reg[10] = 69;

    mem.write(PhysAddr::new(cpu.pc), Op::addu(8, 9, 10));

    // id -> if pipeline overhead
    for _ in 0..2 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    // ex stage
    cpu.run_cycle(&mut mem);
    cpu.advance_cycle();
    // final wb
    cpu.run_cycle(&mut mem);
    cpu.advance_cycle();

    assert_eq!(cpu.reg[8], 42 + 69)
}

#[rstest]
#[instrument]
fn basic_adder_program(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 42;
    cpu.reg[10] = 69;

    cpu.reg[11] = PhysAddr::new(0x1000).as_u32();
    let program = Program::new([
        Op::addu(8, 9, 10),
        Op::sw(8, 11, 0),
        Op::lw(8, 11, 0),
        Op::NOP,
    ]);
    mem.write_all(PhysAddr::new(cpu.pc), program);
    for _ in 0..9 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }
    assert_eq!(cpu.reg[8], 42 + 69)
}
#[rstest]
#[instrument]
fn alu_forwarding_hazard(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 10;
    cpu.reg[10] = 20;

    let program = Program::new([
        Op::addu(8, 9, 10),   // r8 = r9 + r10
        Op::addu(11, 8, 9),   // r11 = r8 + r9
        Op::addu(12, 11, 10), // r12 = r11 + r10
    ]);

    mem.write_all(PhysAddr::new(cpu.pc), program);

    for _ in 0..9 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(cpu.reg[8], 30);
    assert_eq!(cpu.reg[11], 40);
    assert_eq!(cpu.reg[12], 60);
}

#[rstest]
#[instrument]
fn load_use_hazard(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 0x1000;
    cpu.reg[10] = 7;

    // Preload memory
    mem.write(PhysAddr::new(0x1000), 123u32);

    let program = Program::new([
        Op::lw(8, 9, 0), // r8 = Mem[0x1000]
        Op::NOP,
        Op::addu(11, 8, 10), // r11 = r8 + r10
    ]);

    mem.write_all(PhysAddr::new(cpu.pc), program);

    for _ in 0..9 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(cpu.reg[8], 123);
    assert_eq!(cpu.reg[11], 130);
}

#[rstest]
#[instrument]
fn basic_adder_program_2(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 42;
    cpu.reg[10] = 69;

    cpu.reg[11] = PhysAddr::new(0x1000).as_u32();
    let program = Program::new([
        Op::addu(8, 9, 10),
        Op::addi(8, 8, -32),
        Op::sw(8, 11, 0),
        Op::lw(8, 11, 0),
        Op::NOP,
    ]);
    mem.write_all(PhysAddr::new(cpu.pc), program);
    for _ in 0..10 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }
    assert_eq!(cpu.reg[8], 42 + 69 - 32)
}

#[rstest]
#[instrument]
fn basic_adder_program_3(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = 42;
    cpu.reg[10] = 69;

    cpu.reg[11] = PhysAddr::new(0x1000).as_u32();
    let program = Program::new([
        Op::addu(8, 9, 10),
        Op::addi(8, 8, 32),
        Op::sw(8, 11, 0),
        Op::lw(8, 11, 0),
        Op::NOP,
    ]);
    mem.write_all(PhysAddr::new(cpu.pc), program);
    for _ in 0..10 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }
    assert_eq!(cpu.reg[8], 42 + 69 + 32)
}
