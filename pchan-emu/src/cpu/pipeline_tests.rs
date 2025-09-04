use super::*;
use crate::memory::{KSEG0Addr, PhysAddr};
use pchan_utils::setup_tracing;
use pretty_assertions::{assert_eq, assert_matches};
use rstest::*;

#[rstest]
#[instrument]
fn test_pipeline_single_load(setup_tracing: ()) {
    tracing::info!("testing single load");
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    cpu.reg[9] = PhysAddr(0x1000).to_kseg0().as_u32(); // r9 = 0x1000
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

    cpu.reg[8] = KSEG0Addr::from_phys(0x1000).as_u32();
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

    cpu.reg[9] = PhysAddr(0x1000).to_kseg0().as_u32();
    assert!(PhysAddr::try_map(cpu.reg[9]).is_ok());
    assert!(PhysAddr::try_map(cpu.reg[9] + 4).is_ok());
    cpu.reg[10] = PhysAddr(0x2000).to_kseg0().as_u32();
    assert!(PhysAddr::try_map(cpu.reg[10]).is_ok());

    let program = Program::new([
        Op::lw(8, 9, 4),
        Op::NOP,
        Op::lb(9, 8, 1),
        Op::lbu(10, 10, 2),
    ]);
    tracing::info!(%program);
    mem.write_all(PhysAddr::new(0), program);

    mem.write(PhysAddr::new(0x1004), KSEG0Addr::from_phys(0x1234)); // LW source
    tracing::info!("source = 0x{:08X?}", mem.read::<u32>(PhysAddr::new(0x1004)));
    mem.write::<u8>(PhysAddr::new(0x1235), 0x7F); // LB source (0x1235 = r8 + 1)
    mem.write::<u8>(PhysAddr::new(0x2002), 0xFF); // LBU source (r10 + 2)

    // Run enough cycles to complete all three loads
    for _ in 0..15 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(
        cpu.reg[8],
        KSEG0Addr::from_phys(0x1234).as_u32(),
        "LW should load 0x1234 into r8"
    );
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

    cpu.reg[11] = KSEG0Addr::from_phys(0x1000).as_u32();
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

    cpu.reg[9] = KSEG0Addr::from_phys(0x1000).as_u32();
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

    cpu.reg[11] = KSEG0Addr::from_phys(0x1000).as_u32();
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

    cpu.reg[11] = KSEG0Addr::from_phys(0x1000).as_u32();
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

#[rstest]
#[instrument]
fn basic_arithmetic_1(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    let program = Program::new([
        // load values into $r8 and $r9
        Op::addi(8, 0, 12),
        Op::addi(9, 0, 20),
        // perform add into $r10 and sub into $r11
        Op::addu(10, 8, 9),
        Op::subu(11, 9, 8),
        Op::NOP,
    ]);
    mem.write_all(PhysAddr::new(cpu.pc), program);
    for _ in 0..7 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }
    assert_eq!(cpu.reg(10), 12 + 20);
    assert_eq!(cpu.reg(11), 20 - 12);
}

#[rstest]
#[instrument]
fn basic_jump_1(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    let start_pc = cpu.pc;

    // Build a small program:
    // 1. Set $r8 = 42 (will be skipped)
    // 2. Jump to label "target"
    // 3. Set $r8 = 99 (skipped)
    // 4. target: set $r9 = 77
    let program = Program::new([
        Op::addi(8, 0, 42),              // skipped by jump
        Op::j(KSEG0Addr::from_phys(16)), // jump over next instruction
        Op::NOP,
        Op::addi(8, 0, 99), // skipped
        Op::addi(9, 0, 77), // target
        Op::NOP,
    ]);

    // Load program into memor4y
    mem.write_all(PhysAddr::new(start_pc), program);

    // Run enough cycles to execute all instructions
    for _ in 0..12 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    // $r8 should remain 42 (first instruction executed before jump)
    assert_eq!(cpu.reg(8), 42);

    // $r9 should be 77 (instruction after jump target executed)
    assert_eq!(cpu.reg(9), 77);
}

#[rstest]
#[instrument]
fn basic_jump_hazard_1(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    let start_pc = cpu.pc;

    tracing::info!(jump_address = ?KSEG0Addr::from_phys(12));

    let program = Program::new([
        Op::addi(8, 0, 42),              // skipped by jump
        Op::j(KSEG0Addr::from_phys(12)), // jump over next instruction
        Op::addi(8, 0, 99),              // runs anyways
        Op::addi(9, 0, 77),              // target
        Op::NOP,
    ]);

    // Load program into memor4y
    mem.write_all(PhysAddr::new(start_pc), program);

    // Run enough cycles to execute all instructions
    for _ in 0..12 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(cpu.reg(8), 99);

    assert_eq!(cpu.reg(9), 77);
}

#[rstest]
#[instrument]
fn basic_jal_jr_hazard(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    let start_pc = cpu.pc;

    let function_address = 0x0000_2000;
    tracing::info!(
        "virtual address of function is 0x{:08X}",
        KSEG0Addr::from_phys(function_address).as_u32()
    );
    let program = Program::new([
        Op::jal(KSEG0Addr::from_phys(function_address)), // PC 0: jump to function
        Op::addi(8, 0, 1),                               // PC 4: delay slot of jal
        Op::addi(9, 0, 1),                               // PC 8: skipped
    ]);

    // function
    let function = Program::new([
        Op::addi(8, 0, 42), // PC 24: set $8
        Op::jr(RA),         // PC 28: return to $ra
        Op::addi(9, 9, 99), // PC 32: delay slot of jr
    ]);

    // Load program into memory at start_pc
    mem.write_all(PhysAddr::new(start_pc), program);
    mem.write_all(KSEG0Addr::from_phys(function_address), function);

    // Run enough cycles to execute all instructions
    for _ in 0..9 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    // After returning:
    // $8 = 42 (from function)
    // $9 = 99 (from delay slot of jr)
    assert_eq!(cpu.reg(RA), KSEG0Addr::from_phys(8).as_u32());
    assert_eq!(cpu.reg(8), 42);
    assert_eq!(cpu.reg(9), 99);
}

#[rstest]
#[instrument]
fn basic_jalr_test(setup_tracing: ()) {
    let mut cpu = Cpu::default();
    let mut mem = Memory::default();

    let start_pc = cpu.pc;

    let function_address = 0x0000_2000;
    tracing::info!(
        "virtual address of function is 0x{:08X}",
        KSEG0Addr::from_phys(function_address).as_u32()
    );

    let program = Program::new([
        Op::addiu(10, 0, function_address as i16),
        // jump to address in register 10 and write return address to register 11
        Op::jalr(11, 10),
        Op::addi(8, 0, 1), // runs anyways
        Op::addi(9, 9, 1),
    ]);

    let function = Program::new([Op::addi(8, 0, 42), Op::jr(11), Op::addi(9, 0, 99)]);

    mem.write_all(PhysAddr::new(start_pc), program);
    mem.write_all(KSEG0Addr::from_phys(function_address), function);

    for _ in 0..15 {
        cpu.run_cycle(&mut mem);
        cpu.advance_cycle();
    }

    assert_eq!(cpu.reg(10), function_address);
    assert_eq!(cpu.reg(8), 42);
    assert_eq!(cpu.reg(9), 99 + 1);
}
