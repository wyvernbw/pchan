use crate::bindings::ExternFnTable;
use crate::cpu::gte::GteBindings;
use crate::cpu::{Cpu, RA, SP, program};
#[cfg(target_arch = "aarch64")]
use crate::dynarec_v2::emitters::emit_branch_zero;
use crate::dynarec_v2::emitters::{
    AluImm, Boundary, CopMove, CopRegisterKind, DynarecOp, EitherZero, EmitAddImm16Args, EmitCtx,
    EmitSummary, ShiftImm, either_zero, emit_alu_imm, emit_alu_reg, emit_move_to_cop,
    emit_shift_by_reg, emit_shift_imm, shadow,
};
use crate::dynarec_v2::regalloc::AllocResultStackless;
#[cfg(test)]
use crate::dynarec_v2::{CreateDynarecParams, run_step};
use crate::dynarec_v2::{DynEmitter, Guest};
use crate::{Emu, cpu};
#[cfg(test)]
use core::fmt;
use core::num::NonZeroU8;

use arbitrary_int::traits::Integer;
use enum_dispatch::enum_dispatch;

use pchan_utils::default;
#[cfg(test)]
use rstest::rstest;
use smallbox::SmallBox;

use crate::cpu::ops::prelude::*;
use crate::dynarec_v2::{Dynarec, LoadedReg, Reg};
#[cfg(test)]
use crate::memory;
use crate::memory::ext;
use dynasm::dynasm;
use dynasmrt::DynasmApi;

use super::ScheduledEmitter;
use core::alloc::Allocator;

#[cfg(test)]
#[rstest]
#[case(10, 2, 12)]
fn test_addiu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.write_many(
        0x0,
        &program([addiu(12, 10, b as i16), OpCode::new_with_raw_value(69420)]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

/// rough idea taken from bios spu related code
#[cfg(test)]
// #[rstest]
fn test_addiu_andi_loop() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(7, 0, 16),
            addiu(8, 8, 1),
            andi(8, 8, 0x00ff),
            sltu(1, 8, 7),
            bne(1, 0, -4),
            nop(),
            addiu(10, 0, 1),
            OpCode::HALT,
        ]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    while emu.cpu.gpr[10] != 1 {
        tracing::info!(
            r7 = emu.cpu.gpr[7],
            r8 = emu.cpu.gpr[8],
            r10 = emu.cpu.gpr[10]
        );
        run_step(&mut emu, &mut dynarec);
    }
    assert_eq_hex!(emu.cpu.gpr[8], 16);
    assert_eq_hex!(emu.cpu.gpr[7], 16);
    Ok(())
}

#[cfg(test)]
#[rstest]
#[case(10, 2, 8)]
fn test_subu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.cpu.gpr[11] = b;
    emu.write_many(
        0x0,
        &program([subu(12, 10, 11), OpCode::new_with_raw_value(69420)]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

#[cfg(test)]
#[rstest]
#[case::sb(69u8, sb(10, 11, 2))]
#[case::sh(69u8, sh(10, 11, 2))]
#[case::sw(69u8, sw(10, 11, 2))]
fn test_stores<T>(#[case] value: T, #[case] instr: OpCode) -> color_eyre::Result<()>
where
    T: Copy + fmt::Debug + PartialEq,
    T: const memory::Extend<ext::Zero, Out = u32>,
{
    use crate::Emu;
    use crate::cpu::program;
    use crate::memory::ext;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = ext::zero(value);
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write_many(0x0, &program([instr, OpCode::HALT]));

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 3);
    assert_eq!(emu.cpu.pc, 0x8);
    assert_eq!(
        emu.read_ext::<T, ext::Zero>(0x801ffed0 + 2),
        ext::zero(value)
    );

    tracing::info!("returning from test...");
    Ok(())
}

/// this is a write to BIOS rom which is obviously invalid
/// in the future proper emulation should trigger the exception handler
/// for now we just panic.
#[cfg(test)]
#[rstest]
#[expect(clippy::should_panic_without_expect)]
#[should_panic]
fn test_weird_store() {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0xbfc01a60;
    emu.cpu.gpr[cpu::SP as usize] = 0x801ffee8;
    emu.cpu.gpr[cpu::RA as usize] = 0xbfc06ed4;
    emu.write_many(
        emu.cpu.pc,
        &program([
            addiu(cpu::SP, cpu::SP, -0x0018),
            sw(cpu::RA, cpu::SP, 0x0014),
            nop(),
            OpCode::HALT,
        ]),
    );

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    let result = emu.read::<u32>(0x801ffee4);
    assert_eq!(result, 0xbfc06ed4);
}

#[cfg(test)]
#[rstest]
#[case::lb(69, lb, 69)]
#[case::lb(-1i32 as u32, lb, -1i32 as u32)]
#[case::lbu(69, lbu, 69)]
#[case::lbu(0xFF, lbu, 0xFF)]
#[case::lh(69, lh, 69)]
#[case::lh(-1i32 as u32, lh, -1i32 as u32)]
#[case::lhu(69, lhu, 69)]
#[case::lhu(0xFFFF, lhu, 0xFFFF)]
#[case::lw(69, lw, 69)]
#[case::lw(-1i32 as u32, lw, -1i32 as u32)]
fn test_loads(
    #[case] value: u32,
    #[case] instr: impl Fn(u8, u8, i16) -> OpCode,
    #[case] expected: u32,
) {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write(0x801ffed0 + 2, value);
    emu.write_many(0x0, &program([instr(10, 11, 2), OpCode::HALT]));

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.pc, 0x8);
    assert_eq!(emu.cpu.gpr[10], expected);

    tracing::info!("returning from test...");
}

/// this will always try to load 0xcafe_babe
#[cfg(test)]
#[rstest]
#[case(lhu, 0x0, 0xbabe)]
#[case(lhu, 0x2, 0xcafe)]
fn test_partial_loads(
    #[case] instr: impl Fn(u8, u8, i16) -> OpCode,
    #[case] immediate: i16,
    #[case] expected: u32,
) {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = 0x0;
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write::<u32>(0x801ffed0, 0xcafe_babeu32);
    emu.write_many(0x0, &program([instr(10, 11, immediate), OpCode::HALT]));

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.pc, 0x8);
    assert_eq_hex!(emu.cpu.gpr[10], expected);

    tracing::info!("returning from test...");
}

#[cfg(test)]
#[rstest]
fn test_weird_load_01() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = 0xf;
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write::<u32>(0x801ffcd8, 0x0d);
    emu.write::<u32>(0x801ffcd9, 0x0);
    emu.write::<u32>(0x801ffcda, 0x0);
    emu.write::<u32>(0x801ffcdb, 0x0);
    // 0x801ffc78 + 0x62
    emu.write_many(
        0x0,
        &program([
            lui(11, 0x801f_u16 as i16),
            ori(11, 11, 0xfc78_u16 as i16),
            lh(10, 11, 0x62),
            nop(),
            OpCode::HALT,
        ]),
    );

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.gpr[10], 0x0000);

    tracing::info!("returning from test...");
    Ok(())
}

#[cfg(test)]
#[rstest]
#[case::lb(lb)]
#[case::lbu(lbu)]
#[case::lh(lh)]
#[case::lhu(lhu)]
#[case::lw(lw)]
fn test_load_delay(#[case] instr: impl Fn(u8, u8, i16) -> OpCode) {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[11] = 0x100;
    emu.write(0x100 + 2, 69);
    emu.write_many(
        0x0,
        &program([
            instr(10, 11, 2),
            addiu(12, 10, 420),
            addiu(13, 10, 420),
            OpCode::HALT,
        ]),
    );

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.pc, 0x10);
    assert_eq!(emu.cpu.gpr[10], 69);
    assert_eq!(emu.cpu.gpr[12], 420);
    assert_eq!(emu.cpu.gpr[13], 69 + 420);

    tracing::info!("returning from test...");
}

#[rstest]
#[cfg(test)]
fn test_sll_case0() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([addiu(8, 0, 0x4), sll(9, 8, 0x10), OpCode::HALT]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq_hex!(emu.cpu.gpr[8], 0x4);
    assert_eq_hex!(emu.cpu.gpr[9], 0x4 << 0x10);
    assert_eq_hex!(emu.cpu.gpr[9], emu.cpu.gpr[8] << 0x10);

    Ok(())
}

#[cfg(test)]
#[rstest]
#[case(0x0, 12, 0x1234, 0x12340000)]
#[case(0x5678, 9, 0x1234, 0x12340000)]
#[case(0x0, 0, 0x1234, 0x0)]
fn test_lui(
    #[case] initial: u32,
    #[case] rt: Guest,
    #[case] imm: i16,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if rt != 0 {
        emu.cpu.gpr[rt as usize] = initial;
    }
    emu.write_many(0x0, &program([lui(rt, imm), OpCode::HALT]));
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[rt as usize], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);

    Ok(())
}

#[cfg(test)]
#[rstest]
#[case(0x0, 0x0000_1000)]
fn test_jal(#[case] initial_pc: u32, #[case] jump_imm: u32) {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.write_many(
        initial_pc,
        &program([
            jal(jump_imm as _),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode::HALT,
        ]),
    );
    let new_pc = (jump_imm << 2) + (emu.cpu.pc & 0xf0000000);
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.pc, new_pc);
    assert_eq!(emu.cpu.gpr[cpu::RA as usize], initial_pc + 0x8);
}

#[cfg(test)]
#[rstest]
#[case(0x0, (9, 0x0000_1000))]
fn test_jr(#[case] initial_pc: u32, #[case] rs: (Guest, u32)) {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.write_many(
        initial_pc,
        &program([jr(rs.0), addiu(9, 0, 69), addiu(9, 0, 420), OpCode::HALT]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.pc, rs.1);
}

#[cfg(test)]
#[rstest]
fn test_jr_delay_slot() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(8, 0, 0x20),
            nop(),
            jr(8),
            addu(9, 0, 8),
            nop(),
            OpCode::HALT,
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq_hex!(emu.cpu.gpr[8], 0x20);
    assert_eq_hex!(emu.cpu.gpr[9], 0x20);
    assert_eq_hex!(emu.cpu.gpr[9], emu.cpu.gpr[8]);
    Ok(())
}

#[cfg(test)]
#[rstest]
#[case(0x0, (9, 0x0000_1000), 10)]
fn test_jalr(#[case] initial_pc: u32, #[case] rs: (Guest, u32), #[case] rd: Guest) {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.write_many(
        initial_pc,
        &program([
            jalr(rd, rs.0),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode::HALT,
        ]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.pc, rs.1);
    assert_eq!(emu.cpu.gpr[rd as usize], initial_pc + 0x8);
}

#[cfg(test)]
#[rstest]
// beq
#[case::beq_01(beq, 0x1000, (5, 100), (6, 100), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::beq_02(beq, 0x1000, (5, 100), (6, 200), 0x10, 0x1000 + 0x8)]
#[case::beq_03(beq, 0x2000, (7, 0), (8, 0), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::beq_04(beq, 0x3000, (3, 42), (4, 43), -0x8, 0x3000 + 0x8)]
#[case::beq_05(beq, 0x4000, (1, 0xFFFF), (2, 0xFFFF), 0x0, 0x4000 + 0x4)]
#[case::beq_06(beq, 0x5000, (10, 0xDEADBEEF), (11, 0xDEADBEEF), 0x7FF, 0x5000 + 0x4 + (0x7FF << 2))]
// bne
#[case::bne_01(bne, 0x1000, (5, 100), (6, 200), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bne_02(bne, 0x1000, (5, 100), (6, 100), 0x10, 0x1000 + 0x8)]
#[case::bne_03(bne, 0x2000, (7, 0), (8, 1), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bne_04(bne, 0x3000, (3, 0), (4, 0), -0x8, 0x3000 + 0x8)]
#[case::bne_05(bne, 0x5000, (10, 0xDEADBEEF), (11, 0xDEADBEEE), 0x7FF, 0x5000 + 0x4 + (0x7FF << 2))]
#[case::bne_06(bne, 0x6000, (12, 100), (13, 200), 0x0, 0x6000 + 0x4)]
fn test_branch(
    #[case] instr: impl Fn(Guest, Guest, i16) -> OpCode,
    #[case] initial_pc: u32,
    #[case] rs: (Guest, u32),
    #[case] rt: (Guest, u32),
    #[case] offset: i16,
    #[case] expected_pc: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.cpu.gpr[rt.0 as usize] = rt.1;

    emu.write_many(initial_pc, &program([instr(rs.0, rt.0, offset)]));
    emu.write_many(initial_pc + ext::zero(offset), &program([OpCode::HALT]));

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, expected_pc);

    Ok(())
}

#[cfg(test)]
#[rstest]
// bltz - branch if less than zero
#[case::bltz_01(bltz, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bltz_02(bltz, 0x1000, (5, 0), 0x10, 0x1000 + 0x8)]
#[case::bltz_03(bltz, 0x1000, (5, 100), 0x10, 0x1000 + 0x8)]
#[case::bltz_04(bltz, 0x2000, (7, 0x80000000), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bltz_05(bltz, 0x3000, (3, 0x7FFFFFFF), -0x8, 0x3000 + 0x8)]
#[case::bltz_06(bltz, 0x4000, (1, -42i32 as u32), 0x0, 0x4000 + 0x4)]
// bgez - branch if greater than or equal to zero
#[case::bgez_01(bgez, 0x1000, (5, 0), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bgez_02(bgez, 0x1000, (5, 100), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bgez_03(bgez, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x8)]
#[case::bgez_04(bgez, 0x2000, (7, 0x7FFFFFFF), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bgez_05(bgez, 0x3000, (3, 0x80000000), -0x8, 0x3000 + 0x8)]
#[case::bgez_06(bgez, 0x4000, (1, 0), 0x0, 0x4000 + 0x4)]
// blez - branch if less than or equal to zero
#[case::blez_01(blez, 0x1000, (5, 0), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::blez_02(blez, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::blez_03(blez, 0x1000, (5, 100), 0x10, 0x1000 + 0x8)]
#[case::blez_04(blez, 0x2000, (7, 0x80000000), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::blez_05(blez, 0x3000, (3, 0x7FFFFFFF), -0x8, 0x3000 + 0x8)]
#[case::blez_06(blez, 0x4000, (1, -42i32 as u32), 0x0, 0x4000 + 0x4)]
// bgtz - branch if greater than zero
#[case::bgtz_01(bgtz, 0x1000, (5, 100), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bgtz_02(bgtz, 0x1000, (5, 0), 0x10, 0x1000 + 0x8)]
#[case::bgtz_03(bgtz, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x8)]
#[case::bgtz_04(bgtz, 0x2000, (7, 0x7FFFFFFF), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bgtz_05(bgtz, 0x3000, (3, 0x80000000), -0x8, 0x3000 + 0x8)]
#[case::bgtz_06(bgtz, 0x4000, (1, 1), 0x0, 0x4000 + 0x4)]
fn test_branch_zero(
    #[case] instr: impl Fn(Guest, i16) -> OpCode,
    #[case] initial_pc: u32,
    #[case] rs: (Guest, u32),
    #[case] offset: i16,
    #[case] expected_pc: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;

    emu.write_many(initial_pc, &program([instr(rs.0, offset)]));
    emu.write_many(initial_pc + ext::zero(offset), &program([OpCode::HALT]));

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, expected_pc);

    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_store_loop() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(9, 0, 69),
            addiu(10, 0, 0x0000_1000),
            addiu(11, 0, 0x0000_2000),
            sw(9, 10, 0),
            bne(10, 11, -2),
            addiu(10, 10, 0x4),
            OpCode::HALT,
        ]),
    );

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    loop {
        run_step(&mut emu, &mut dynarec);
        assert!(emu.cpu.gpr[10] <= emu.cpu.gpr[11] + 0x8);

        if emu.cpu.pc == 24 {
            break;
        }
    }

    tracing::info!(?emu.cpu);

    Ok(())
}

#[cfg(test)]
#[rstest]
#[case::div(div, (8, 10), (9, 2), 0x00000000_00000005)]
#[case::div(div, (0, 0), (8, 3), 0x00000000_00000000)]
#[case::div(div, (8, 2), (9, 0), 0x00000002_ffffffff)]
#[case::div(div, (8, -2i32 as u32), (9, 0), 0xfffffffe_00000001)]
#[case::div(div, (8, -10i32 as u32), (9, 2), 0x00000000_fffffffb)]
#[case::divu(divu, (8, 10), (9, 2), 0x00000000_00000005)]
#[case::divu(divu, (8, 2), (9, 0), 0x00000002_ffffffff)]
#[case::mult(mult, (8, 2), (9, 15), 30)]
#[case::mult(mult, (8, 2), (9, -15i32 as u32), -30i32 as u64)]
#[case::multu(multu, (8, 2), (9, 15), 30)]
/// form: {inst} ${reg1}={value1}, ${reg2}={value2} ; assert hilo = {expected}
pub fn test_mul_div(
    #[case] instr: impl Fn(u8, u8) -> OpCode,
    #[case] rs: (Guest, u32),
    #[case] rt: (Guest, u32),
    #[case] expected: u64,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::{hex, setup_tracing};

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.hilo = 0xDEAD_BEEF_DEAD_BEEF;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.cpu.gpr[rt.0 as usize] = rt.1;
    assert_eq!(emu.cpu.gpr[0], 0);

    emu.write_many(0x0, &program([instr(rs.0, rt.0), OpCode::HALT]));
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    assert_eq!(emu.cpu.hilo, expected);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

#[cfg(test)]
#[rstest]
#[case(mfhi, 0xDEAD_BEEF_1234_5678, 0xDEAD_BEEF)]
#[case(mflo, 0xDEAD_BEEF_1234_5678, 0x1234_5678)]
pub fn test_mfhilo(
    #[case] instr: impl Fn(u8) -> OpCode,
    #[case] hilo: u64,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::{hex, setup_tracing};

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.hilo = hilo;
    emu.write_many(0x0, &program([instr(9), OpCode::HALT]));
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    tracing::info!(r = %hex(emu.cpu.gpr[9]));
    assert_eq!(emu.cpu.hilo, hilo);
    assert_eq!(emu.cpu.gpr[9], expected);
    Ok(())
}

#[cfg(test)]
#[rstest]
pub fn test_load_0xbfc01a78() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(0x0, &program([OpCode::HALT]));
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!(?emu.cpu);
    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_branch_and_store() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(7, 7, 0x200),
            addiu(8, 8, 0x12),
            beq(9, 0, 0x8),
            sw(8, 7, 0),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    assert_eq_hex!(emu.read::<u32>(0x200), 0x12);

    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_0x8004f454_move_in_jump_delay() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.gpr[cpu::SP as usize] = 0x801ffd50;
    emu.cpu.gpr[4] = 0x12;
    emu.write_many(
        0x0,
        &program([
            OpCode::HALT,
            addiu(cpu::SP, cpu::SP, 0x4),
            sw(16, cpu::SP, 0x0018),
            addu(16, 0, 4),
            sw(cpu::RA, cpu::SP, 0x001c),
            addiu(3, 16, 0x001c),
            addu(4, 3, 0),
            sw(3, cpu::SP, 0x0024),
            jal(0x0),
            addu(5, 0, 16),
        ]),
    );
    emu.cpu.pc = 0x4;
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    assert_eq_hex!(emu.cpu.gpr[5], 0x12);

    Ok(())
}

#[cfg(test)]
#[rstest]
#[case(0, 5, 10)]
#[case(2, 5, 10)]
#[case(2, 31, 10)]
fn test_lwcn(#[case] cop: u8, #[case] rt: u8, #[case] rs: u8) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[rs as usize] = 0x100;
    emu.write(0x100, 0xcafebabe_u32);

    emu.write_many(0x0, &program([lwcn(cop, rt, rs, 0x0), OpCode::HALT]));

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);
    tracing::info!(?emu.cpu);
    match cop {
        0 => assert_eq!(emu.cpu.cop0.reg[rt as usize], 0xcafebabe),
        2 => assert_eq!(emu.cpu.gte.data.as_slice_mut()[rt as usize], 0xcafebabe),
        _ => panic!("get out"),
    }

    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_bltzal_bgezal() {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    {
        let mut emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([addiu(8, 0, -10), bltzal(8, 0x100), OpCode::HALT]),
        );
        let mut dynarec = Dynarec::new(CreateDynarecParams::default());
        run_step(&mut emu, &mut dynarec);
        tracing::info!(?emu.cpu);
        assert_eq_hex!(emu.cpu.gpr[8] as i32, -10);
        assert_eq_hex!(emu.cpu["$ra"], 0xc);
        assert_eq_hex!(emu.cpu.pc, 0x408);
    }

    {
        let mut emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([addiu(8, 0, 10), bgezal(8, 0x100), OpCode::HALT]),
        );
        let mut dynarec = Dynarec::new(CreateDynarecParams::default());
        run_step(&mut emu, &mut dynarec);
        tracing::info!(?emu.cpu);
        assert_eq_hex!(emu.cpu.gpr[8] as i32, 10);
        assert_eq_hex!(emu.cpu["$ra"], 0xc);
        assert_eq_hex!(emu.cpu.pc, 0x408);
    }
}

#[cfg(test)]
#[rstest]
fn test_gte_register_moves() {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();

    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.set_reg(8, 0xc31b9378);
    emu.set_reg(9, 0x7fffffff);
    #[rustfmt::skip]
    emu.write_many(
        0x0,
        &program([
            mtcn(2, 8, 0),
            mtcn(2, 9, 1),
            nop(),
            nop(),
            mfcn(2, 10, 0),
            mfcn(2, 11, 1),
            nop(),
            nop(),
            OpCode::HALT
        ]),
    );
    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    smol::block_on(run_step(&mut emu, &mut dynarec));
    tracing::info!(?emu.cpu);
    assert_eq_hex!(emu.cpu.gpr[10], 0xc31b9378);
    assert_eq_hex!(emu.cpu.gpr[11], 0x7fffffff);
}

const fn load_par_program_one_imm(imm: i16) -> [u32; 2] {
    program([lwl(9, SP, imm), lwr(10, SP, imm)])
}

const fn load_seq_program_one_imm(imm: i16) -> [u32; 2] {
    program([lwl(9, SP, imm), lwr(9, SP, imm)])
}

const fn load_seq_two_imm(a: i16, b: i16) -> [u32; 2] {
    program([lwl(9, SP, a), lwr(9, SP, b)])
}

#[rstest]
#[case(load_par_program_one_imm(0x0), 0x00ff_ffff, 0x0302_0100)]
#[case(load_par_program_one_imm(0x1), 0x0100_ffff, 0xff03_0201)]
#[case(load_par_program_one_imm(0x2), 0x0201_00ff, 0xffff_0302)]
#[case(load_par_program_one_imm(0x3), 0x0302_0100, 0xffff_ff03)]
#[case(load_par_program_one_imm(0x4), 0x04ff_ffff, 0x0706_0504)]
fn test_par_lwl_lwr<const N: usize>(
    #[case] prog: [u32; N],
    #[case] t1: u32,
    #[case] t2: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::dynarec_v2::{CreateDynarecParams, Dynarec, run_step};
    use assert_hex::assert_eq_hex;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu["$sp"] = 0x8000_00f0;
    emu.cpu.gpr[9] = 0xffff_ffff;
    emu.cpu.gpr[10] = 0xffff_ffff;
    for i in 0x0..0x10 {
        emu.write(emu.cpu["$sp"] + i, i);
    }
    let prog = [prog.as_slice(), [OpCode::HALT.raw_value()].as_slice()].concat();
    emu.write_many(0x0, &prog);

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.gpr[9], t1);
    assert_eq_hex!(emu.cpu.gpr[10], t2);

    Ok(())
}

#[rstest]
#[case(load_seq_program_one_imm(0x0), 0x0302_0100)]
#[case(load_seq_program_one_imm(0x1), 0x0103_0201)]
#[case(load_seq_program_one_imm(0x2), 0x0201_0302)]
#[case(load_seq_program_one_imm(0x3), 0x0302_0103)]
#[case(load_seq_program_one_imm(0x4), 0x0706_0504)]
// ulw
#[case(load_seq_two_imm(0x3, 0x0), 0x0302_0100)]
#[case(load_seq_two_imm(0x4, 0x1), 0x0403_0201)]
#[case(load_seq_two_imm(0x5, 0x2), 0x0504_0302)]
#[case(load_seq_two_imm(0x6, 0x3), 0x0605_0403)]
fn test_seq_lwl_lwr<const N: usize>(
    #[case] prog: [u32; N],
    #[case] t1: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::dynarec_v2::{CreateDynarecParams, Dynarec, run_step};
    use assert_hex::assert_eq_hex;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu["$sp"] = 0x8000_00f0;
    emu.cpu.gpr[9] = 0xffff_ffff;
    for i in 0x0..0x10 {
        emu.write(emu.cpu["$sp"] + i, i);
    }
    let prog = [prog.as_slice(), [OpCode::HALT.raw_value()].as_slice()].concat();
    emu.write_many(0x0, &prog);

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.gpr[9], t1);

    Ok(())
}

fn single_write_program(imm: i16, op: impl const Fn(u8, u8, i16) -> OpCode) -> [u32; 3] {
    program([lui(9, 0x0302), ori(9, 9, 0x0100), op(9, SP, imm)])
}

fn two_writes_program(imm: i16) -> [u32; 4] {
    program([
        lui(9, 0x0302),
        ori(9, 9, 0x0100),
        swl(9, SP, imm),
        swr(9, SP, imm),
    ])
}

/// Tests for the `lwl`, `lwr`, `swl` and `swr` instructions.
///
/// The tests assume the following memory setup:
/// - `[0x0($sp)] = 0x0`
/// - `[0x1($sp)] = 0x1`
/// - `[0x2($sp)] = 0x2`
/// - etc.
///
/// # Load tests
///
/// Load tests will be of the form:
///
/// ```asm
/// lwl $t1,imm($sp)
/// lwr $t2,imm($sp)
/// ```
#[rstest]
#[case(single_write_program(0x0, swl), 0xffff_ff03)]
#[case(single_write_program(0x1, swl), 0xffff_0302)]
#[case(single_write_program(0x2, swl), 0xff03_0201)]
#[case(single_write_program(0x3, swl), 0x0302_0100)]
#[case(single_write_program(0x4, swl), 0xffff_ffff)]
#[case(single_write_program(0x0, swr), 0x0302_0100)]
#[case(single_write_program(0x1, swr), 0x0201_00ff)]
#[case(single_write_program(0x2, swr), 0x0100_ffff)]
#[case(single_write_program(0x3, swr), 0x00ff_ffff)]
#[case(single_write_program(0x4, swr), 0xffff_ffff)]
#[case(two_writes_program(0x0), 0x0302_0100)]
#[case(two_writes_program(0x1), 0x0201_0002)]
#[case(two_writes_program(0x2), 0x0100_0201)]
#[case(two_writes_program(0x3), 0x0002_0100)]
fn test_single_write<const N: usize>(
    #[case] prog: [u32; N],
    #[case] word: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::dynarec_v2::{CreateDynarecParams, Dynarec, run_step};
    use assert_hex::assert_eq_hex;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu["$sp"] = 0x8000_00f0;
    for i in 0x0..0x10 {
        emu.write::<u8>(emu.cpu["$sp"] + i, 0xff);
    }
    let prog = [prog.as_slice(), [OpCode::HALT.raw_value()].as_slice()].concat();
    emu.write_many(0x0, &prog);

    let mut dynarec = Dynarec::new(CreateDynarecParams::default());
    run_step(&mut emu, &mut dynarec);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    let written = emu.read::<u32>(emu.cpu["$sp"]);
    assert_eq_hex!(written, word);

    Ok(())
}
