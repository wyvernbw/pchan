#![cfg(test)]

use crate::Emu;
use crate::cpu::ops::*;
use crate::cpu::program;
use crate::dynarec_v2::regalloc::Guest;
use pchan_utils::setup_tracing;
use rstest::{fixture, rstest};

use crate::run::{Runner, RunnerConfig, RunnerMode};

#[fixture]
fn interp() -> Runner {
    Runner::new().with_config(RunnerConfig {
        force_mode: Some(RunnerMode::Interpreter),
    })
}
#[fixture]
fn dynarec() -> Runner {
    Runner::new().with_config(RunnerConfig {
        force_mode: Some(RunnerMode::Dynarec),
    })
}

#[rstest]
fn test_bltzal_bgezal(#[values(dynarec(), interp())] mut runner: Runner) {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::emitters::DecodedOp;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    tracing::info!(bltzal =?DecodedOp::new(bltzal(8, 0x100)));
    {
        let mut emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([addiu(8, 0, -10), bltzal(8, 0x100), OpCode::HALT]),
        );
        runner.execute(&mut emu);
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
        runner.execute(&mut emu);
        tracing::info!(?emu.cpu);
        assert_eq_hex!(emu.cpu.gpr[8] as i32, 10);
        assert_eq_hex!(emu.cpu["$ra"], 0xc);
        assert_eq_hex!(emu.cpu.pc, 0x408);
    }
}

#[rstest]
fn load_delay_01(#[values(dynarec(), interp())] mut runner: Runner) {
    use crate::cpu::program;
    setup_tracing();

    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(8, 0, 69),
            sw(8, 0, 0x100),
            nop(),
            lw(9, 0, 0x100),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    runner.execute(&mut emu);
    assert_eq!(emu.get_reg(9), 69)
}

#[rstest]
fn load_delay_02(#[values(dynarec(), interp())] mut runner: Runner) {
    setup_tracing();
    let mut emu: Emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(8, 0, 69),
            sw(8, 0, 0x100),
            lw(9, 0, 0x100),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    runner.execute(&mut emu);
    assert_eq!(emu.get_reg(9), 69)
}

#[rstest]
fn load_delay_03(#[values(dynarec(), interp())] mut runner: Runner) {
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu: Emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(8, 0, 0x100),
            addiu(9, 0, 69),
            sw(8, 0, 0x100),
            sw(9, 0, 0x200),
            nop(),
            addiu(9, 0, 0),
            lw(9, 0, 0x100),
            nop(),
            lw(9, 9, 0x100),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    runner.execute(&mut emu);
    assert_eq!(emu.read::<u32>(0x200), 69, "expected 69 in memory");
    assert_eq!(emu.get_reg(9), 69, "expected 69 in register")
}

#[rstest]
#[case::div_00(div, (8, 10), (9, 2), 0x00000000_00000005)]
#[case::div_01(div, (0, 0), (8, 3), 0x00000000_00000000)]
#[case::div_02(div, (8, 2), (9, 0), 0x00000002_ffffffff)]
#[case::div_03(div, (8, -2i32 as u32), (9, 0), 0xfffffffe_00000001)]
#[case::div_04(div, (8, -10i32 as u32), (9, 2), 0x00000000_fffffffb)]
#[case::divu_00(divu, (8, 10), (9, 2), 0x00000000_00000005)]
#[case::divu_01(divu, (8, 2), (9, 0), 0x00000002_ffffffff)]
#[case::mult_00(mult, (8, 2), (9, 15), 30)]
#[case::mult_01(mult, (8, 2), (9, -15i32 as u32), -30i32 as u64)]
#[case::multu(multu, (8, 2), (9, 15), 30)]
/// form: {inst} ${reg1}={value1}, ${reg2}={value2} ; assert hilo = {expected}
pub fn test_mul_div(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] instr: impl Fn(u8, u8) -> OpCode,
    #[case] rs: (u8, u32),
    #[case] rt: (u8, u32),
    #[case] expected: u64,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.hilo = 0xDEAD_BEEF_DEAD_BEEF;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.cpu.gpr[rt.0 as usize] = rt.1;
    assert_eq!(emu.cpu.gpr[0], 0);

    emu.write_many(
        0x0,
        &program([instr(rs.0, rt.0), nop(), nop(), OpCode::HALT]),
    );
    runner.execute(&mut emu);

    assert_eq_hex!(emu.cpu.hilo, expected);
    assert_eq_hex!(emu.cpu.pc, 16);
    Ok(())
}

#[rstest]
pub fn test_la(#[values(dynarec(), interp())] mut runner: Runner) {
    let mut emu: Emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([lui(8, 0), addiu(8, 8, 0x0c80), nop(), nop(), OpCode::HALT]),
    );
    runner.execute(&mut emu);
    assert_eq!(emu.get_reg(8), 0x0c80)
}

#[rstest]
#[case::subu_01(subu, (12, 93), (10, 100), (11, 7))]
#[case::subu_02(subu, (12, 100), (10, 100), (0, 0))]
#[case::addu_01(addu, (12, 21), (10, 19), (11, 2))]
#[case::addu_02(addu, (0, 0), (10, 69), (11, 15))]
#[case::addu_03(addu, (12, 0), (0, 0), (0, 0))]
#[case::addu_04(addu, (12, 15), (10, 15), (0, 0))]
#[case::and_01(and, (12, 0b1010), (10, 0b1111), (11, 0b1010))]
#[case::and_02(and, (12, 0), (0, 0), (11, 0b1010))]
#[case::and_03(and, (12, 0), (11, 0b1010), (0, 0))]
#[case::and_04(and, (12, 0), (0, 0), (0, 0))]
#[case::or_01(or, (12, 0b1111), (10, 0b1010), (11, 0b0101))]
#[case::or_02(or, (12, 0b1010), (10, 0b1010), (0, 0))]
#[case::or_03(or, (12, 0b0101), (0, 0), (11, 0b0101))]
#[case::or_04(or, (12, 0), (0, 0), (0, 0))]
#[case::or_05(or, (12, 0xFFFFFFFF), (10, 0xAAAAAAAA), (11, 0x55555555))]
#[case::xor_01(xor, (12, 0b0101), (10, 0b1111), (11, 0b1010))]
#[case::xor_02(xor, (12, 0b1010), (10, 0b1010), (0, 0))]
#[case::xor_03(xor, (12, 0b0101), (0, 0), (11, 0b0101))]
#[case::xor_04(xor, (12, 0), (0, 0), (0, 0))]
#[case::xor_05(xor, (12, 0), (10, 0xAAAAAAAA), (11, 0xAAAAAAAA))]
#[case::xor_06(xor, (12, 0xFFFFFFFF), (10, 0xAAAAAAAA), (11, 0x55555555))]
#[case::nor_01(nor, (12, !0b1111), (10, 0b1010), (11, 0b0101))]
#[case::nor_02(nor, (12, !0b1010), (10, 0b1010), (0, 0))]
#[case::nor_03(nor, (12, !0b0101), (0, 0), (11, 0b0101))]
#[case::nor_04(nor, (12, 0xFFFFFFFF), (0, 0), (0, 0))]
#[case::nor_05(nor, (12, 0), (10, 0xAAAAAAAA), (11, 0x55555555))]
#[case::nor_06(nor, (12, 0x55555555), (10, 0xAAAAAAAA), (11, 0xAAAAAAAA))]
#[case::sllv_01(sllv, (12, 0b10100), (10, 0b101), (11, 2))]
#[case::sllv_02(sllv, (12, 0b101), (10, 0b101), (11, 0))]
#[case::sllv_03(sllv, (12, 0), (0, 0), (11, 5))]
#[case::sllv_04(sllv, (12, 0), (0, 0), (0, 0))]
#[case::sllv_05(sllv, (12, 0x80000000), (10, 1), (11, 31))]
#[case::sllv_06(sllv, (12, 0x5555_5500), (10, 0xAAAA_AAAA), (11, 7))]
#[case::srlv_01(srlv, (12, 0b010), (10, 0b10100), (11, 3))]
#[case::srlv_02(srlv, (12, 0b101), (10, 0b101), (11, 0))]
#[case::srlv_03(srlv, (12, 0), (10, 0), (11, 5))]
#[case::srlv_04(srlv, (12, 0), (10, 0), (11, 0))]
#[case::srlv_05(srlv, (12, 1), (10, 0x80000000), (11, 31))]
#[case::srlv_06(srlv, (12, 0x0155_5555), (10, 0xAAAA_AAAA), (11, 7))]
#[case::srav_01(srav, (12, 0b11111111111111111111111111111110), (10, 0b11111111111111111111111111110100u32 as i32 as u32), (11, 3))]
#[case::srav_02(srav, (12, 0b101), (10, 0b101), (11, 0))]
#[case::srav_03(srav, (12, 0), (10, 0), (11, 5))]
#[case::srav_04(srav, (12, 0), (10, 0), (11, 0))]
#[case::srav_05(srav, (12, 0xFFFFFFFF), (10, 0x80000000), (11, 31))]
#[case::srav_06(srav, (12, 0xFF555555), (10, 0xAAAA_AAAA), (11, 7))]
fn test_alu_reg(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] instr: impl Fn(u8, u8, u8) -> OpCode,
    #[case] expected: (Guest, u32),
    #[case] a: (Guest, u32),
    #[case] b: (Guest, u32),
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if expected.0 != 0 {
        emu.cpu.gpr[expected.0 as usize] = 1231123;
    }
    emu.cpu.gpr[a.0 as usize] = a.1;
    emu.cpu.gpr[b.0 as usize] = b.1;
    emu.write_many(0x0, &program([instr(expected.0, a.0, b.0), OpCode::HALT]));
    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[expected.0 as usize], expected.1);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

#[rstest]
#[case::sll_01(sll, (12, 0b10100), (10, 0b101), 2)]
#[case::sll_02(sll, (12, 0b101), (10, 0b101), 0)]
#[case::sll_03(sll, (12, 0), (0, 0), 5)]
#[case::sll_04(sll, (12, 0), (0, 0), 0)]
#[case::sll_05(sll, (12, 0x80000000), (10, 1), 31)]
#[case::sll_06(sll, (12, 0x5555_5500), (10, 0xAAAA_AAAA), 7)]
#[case::srl_01(srl, (12, 0b010), (10, 0b10100), 3)]
#[case::srl_02(srl, (12, 0b101), (10, 0b101), 0)]
#[case::srl_03(srl, (12, 0), (10, 0), 5)]
#[case::srl_04(srl, (12, 0), (10, 0), 0)]
#[case::srl_05(srl, (12, 1), (10, 0x80000000), 31)]
#[case::srl_06(srl, (12, 0x0155_5555), (10, 0xAAAA_AAAA), 7)]
#[case::sra_01(sra, (12, 0b11111111111111111111111111111110), (10, 0b11111111111111111111111111110100u32 as i32 as u32), 3)]
#[case::sra_02(sra, (12, 0b101), (10, 0b101), 0)]
#[case::sra_03(sra, (12, 0), (10, 0), 5)]
#[case::sra_04(sra, (12, 0), (10, 0), 0)]
#[case::sra_05(sra, (12, 0xFFFFFFFF), (10, 0x80000000), 31)]
#[case::sra_06(sra, (12, 0xFF555555), (10, 0xAAAA_AAAA), 7)]
#[case::andi_01(andi, (12, 0b1010), (10, 0b1111), 0b1010i16)]
#[case::andi_02(andi, (12, 0), (0, 0), 0b1010i16)]
#[case::andi_03(andi, (12, 0), (11, 0b1010), 0i16)]
#[case::andi_04(andi, (12, 0), (0, 0), 0i16)]
#[case::ori_01(ori, (12, 0b1111), (10, 0b1010), 0b0101)]
#[case::ori_02(ori, (12, 0b1010), (10, 0b1010), 0)]
#[case::ori_03(ori, (12, 0b0101), (0, 0), 0b0101)]
#[case::ori_04(ori, (12, 0), (0, 0), 0)]
#[case::ori_05(ori, (12, 0xAAAAFFFF), (10, 0xAAAAAAAA), 0xFFFFu16 as i16)]
#[case::xori_01(xori, (12, 0b0101), (10, 0b1111), 0b1010)]
#[case::xori_02(xori, (12, 0b1010), (10, 0b1010), 0)]
#[case::xori_03(xori, (12, 0b0101), (0, 0), 0b0101)]
#[case::xori_04(xori, (12, 0), (0, 0), 0)]
#[case::xori_05(xori, (12, 0xAAAA5555), (10, 0xAAAAAAAA), 0xFFFFu16 as i16)]
#[case::xori_06(xori, (12, 0xAAAAFFFF), (10, 0xAAAAAAAA), 0x5555)]
fn test_alu_imm<I: Into<i16>>(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] instr: impl Fn(u8, u8, I) -> OpCode,
    #[case] expected: (Guest, u32),
    #[case] a: (Guest, u32),
    #[case] b: I,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if expected.0 != 0 {
        emu.cpu.gpr[expected.0 as usize] = 1231123;
    }
    emu.cpu.gpr[a.0 as usize] = a.1;
    emu.write_many(0x0, &program([instr(expected.0, a.0, b), OpCode::HALT]));
    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[expected.0 as usize], expected.1);
    assert_eq!(emu.cpu.pc, 0x8);

    Ok(())
}

#[rstest]
#[case(0, 5, 10)]
#[case(2, 5, 10)]
#[case(2, 31, 10)] // really pushing it
fn test_mtcn(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] cop: u8,
    #[case] rd: u8,
    #[case] rt: u8,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[rt as usize] = 69;

    emu.write_many(
        0x0,
        &program([mtcn(cop, rt, rd), nop(), nop(), OpCode::HALT]),
    );

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    match cop {
        0 => assert_eq!(emu.cpu.cop0.reg[rd as usize], 69),
        2 => assert_eq!(emu.cpu.cop2.reg[rd as usize], 69),
        _ => panic!("get out"),
    }

    Ok(())
}

#[rstest]
#[case(0, 5, 10)]
#[case(2, 5, 10)]
#[case(2, 31, 10)] // really pushing it
fn test_mfcn(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] cop: u8,
    #[case] rd: u8,
    #[case] rt: u8,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.set_cop(cop, rd, 69);

    emu.write_many(
        0x0,
        &program([mfcn(cop, rt, rd), nop(), nop(), OpCode::HALT]),
    );

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.get_reg(rt), 69);

    Ok(())
}
#[rstest]
#[case(2, 5, 10)]
#[case(2, 31, 10)]
fn test_ctcn(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] cop: u8,
    #[case] rd: u8,
    #[case] rt: u8,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[rt as usize] = 69;

    emu.write_many(
        0x0,
        &program([ctcn(cop, rt, rd), nop(), nop(), OpCode::HALT]),
    );

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    match cop {
        2 => assert_eq!(emu.cpu.cop2.reg[rd as usize + 32], 69),
        _ => panic!("get out"),
    }

    Ok(())
}

#[rstest]
fn test_mtcn_enable_isc(
    #[values(dynarec(), interp())] mut runner: Runner,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([lui(9, 0x0001), mtcn(0, 9, 12), nop(), nop(), OpCode::HALT]),
    );

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.cop0.reg[12], 0x0001_0000);
    assert!(emu.cpu.isc());

    Ok(())
}

#[rstest]
fn test_mtcn_enable_irq(
    #[values(dynarec(), interp())] mut runner: Runner,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::{hex, setup_tracing};

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(9, 9, 0x0401),
            mtcn(0, 9, 12),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.cop0.reg[12], 0x0000_0401);
    assert!(emu.cpu.cop0.status().iec());
    assert!(emu.cpu.cop0.status().irq_mask(2));

    emu.raise_irq_exception();
    tracing::info!(irq_mask = %hex(emu.cpu.cop0.status().irq_mask_combined()));
    tracing::info!(irq_pending  = %hex(emu.cpu.cop0.cause().irq_pending_combined()));
    tracing::info!(iec = emu.cpu.cop0.status().iec());

    {
        let sr = emu.cpu.cop0.status();
        let cause = emu.cpu.cop0.cause();
        assert!(cause.irq_pending_combined() & sr.irq_mask_combined() != 0 && sr.iec());
    }

    emu.run_io();

    Ok(())
}

#[rstest]
#[case(mthi, (9, 0xdeadbeef), 0xdeadbeef_00000000)]
#[case(mtlo, (9, 0xdeadbeef), 0x00000000_deadbeef)]
pub fn test_mthilo(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] instr: impl Fn(u8) -> OpCode,
    #[case] (rs, rs_value): (u8, u32),
    #[case] expected: u64,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::{hex, setup_tracing};

    setup_tracing();
    let mut emu = Emu::default();

    if rs != 0 {
        emu.cpu.gpr[rs as usize] = rs_value;
    }
    emu.write_many(0x0, &program([instr(rs), nop(), nop(), OpCode::HALT]));
    runner.execute(&mut emu);

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    assert_eq!(emu.cpu.hilo, expected);
    Ok(())
}

/// Note: After accessing the lo/hi registers, there seems to be a strange rule
/// that one should not touch the lo/hi registers in the next 2 cycles or so... not
/// yet understood if/when/how that rule applies...?
#[cfg(false)]
#[rstest]
pub fn test_mthi_mfhi(#[values(dynarec(), interp())] mut runner: Runner) -> color_eyre::Result<()> {
    use pchan_utils::hex;

    use crate::Emu;
    use crate::cpu::program;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(9, 0, 69),
            mthi(9),
            nop(),
            mfhi(10),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    runner.execute(&mut emu);

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    assert_ne!(emu.cpu.hilo, 0);
    assert_ne!(emu.cpu.gpr[10], 69);
    assert_eq!(emu.cpu.gpr[10], 0);

    // correct version:

    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(9, 0, 69),
            mthi(9),
            nop(),
            nop(),
            mfhi(10),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    runner.execute(&mut emu);

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    assert_ne!(emu.cpu.hilo, 0);
    assert_eq!(emu.cpu.gpr[10], 69);

    Ok(())
}

#[rstest]
#[case(0x0, 0x0000_1000)]
fn test_j(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] initial_pc: u32,
    #[case] jump_imm: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.write_many(
        initial_pc,
        &program([
            j(jump_imm as _),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode::HALT,
            nop(),
            nop(),
        ]),
    );
    let new_pc = (jump_imm << 2) + (emu.cpu.pc & 0xf0000000);
    emu.write(new_pc, OpCode::HALT);
    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    match runner.config.force_mode.unwrap() {
        RunnerMode::Dynarec => {
            assert_eq!(emu.cpu.pc, new_pc);
        }
        RunnerMode::Interpreter => {
            assert_eq!(emu.cpu.pc, new_pc + 4);
        }
    };

    Ok(())
}

#[rstest]
fn test_branch_and_store(
    #[values(dynarec(), interp())] mut runner: Runner,
) -> color_eyre::Result<()> {
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
            beq(9, 0, 0x2),
            sw(8, 7, 0),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );

    runner.execute(&mut emu);

    assert_eq_hex!(emu.read::<u32>(0x200), 0x12);

    Ok(())
}

#[rstest]
fn test_0x8004f454_move_in_jump_delay(
    #[values(dynarec(), interp())] mut runner: Runner,
) -> color_eyre::Result<()> {
    use crate::cpu::program;
    use crate::{Emu, cpu};
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.gpr[cpu::SP as usize] = 0x801ffd50;
    emu.cpu.gpr[4] = 0x12;
    emu.cpu.pc = 0x4;
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

    runner.execute(&mut emu);

    assert_eq_hex!(emu.cpu.gpr[5], 0x12);

    Ok(())
}

#[rstest]
fn test_weird_load_01(#[values(dynarec(), interp())] mut runner: Runner) -> color_eyre::Result<()> {
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
            nop(),
            OpCode::HALT,
        ]),
    );

    runner.execute(&mut emu);

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.gpr[10], 0x0000);

    tracing::info!("returning from test...");
    Ok(())
}

#[rstest]
#[case(0, 5, 10)]
#[case(2, 5, 10)]
#[case(2, 31, 10)]
fn test_lwcn(
    #[values(dynarec(), interp())] mut runner: Runner,
    #[case] cop: u8,
    #[case] rt: u8,
    #[case] rs: u8,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[rs as usize] = 0x100;
    emu.write(0x100, 0xcafebabe_u32);

    emu.write_many(
        0x0,
        &program([lwcn(cop, rt, rs, 0x0), nop(), nop(), OpCode::HALT]),
    );

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.get_cop(cop, rt), 0xcafebabe);

    Ok(())
}

// DONE: on interpreter, consecutive branches are additive ;-;
#[rstest]
// #[cfg(false)]
fn test_branch_in_branch_delay_slot(#[values(dynarec(), interp())] mut runner: Runner) {
    use crate::cpu::program;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x4;
    emu.write_many(
        0x0,
        &program([OpCode::HALT, nop(), beq(0, 0, -3), beq(0, 0, 10)]),
    );
    emu.write(56, OpCode::HALT);

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, 56);

    let mut emu = Emu::default();
    emu.cpu.pc = 0x4;
    emu.write_many(
        0x0,
        &program([OpCode::HALT, nop(), beq(0, 0, -3), jal(0x100)]),
    );
    emu.write(0x100 << 2, OpCode::HALT);

    runner.execute(&mut emu);
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, 0x100 << 2);
    assert_ne!(emu.cpu["$ra"], 0x0, "should have saved $ra");
}

mod test_unaligned_load_stores {
    //! Tests for the `lwl`, `lwr`, `swl` and `swr` instructions.
    //!
    //! The tests assume the following memory setup:
    //! - `[0x0($sp)] = 0x0`
    //! - `[0x1($sp)] = 0x1`
    //! - `[0x2($sp)] = 0x2`
    //! - etc.
    //!
    //! # Load tests
    //!
    //! Load tests will be of the form:
    //!
    //! ```asm
    //! lwl $t1,imm($sp)
    //! lwr $t2,imm($sp)
    //! ```

    use rstest::rstest;

    use crate::cpu::cpu_tests::{dynarec, interp};
    use crate::cpu::ops::{OpCode, lui, lwl, lwr, nop, ori, swl, swr};
    use crate::cpu::{SP, program};
    use crate::run::Runner;

    const fn load_par_program_one_imm(imm: i16) -> [u32; 4] {
        program([lwl(9, SP, imm), lwr(10, SP, imm), nop(), nop()])
    }

    const fn load_seq_program_one_imm(imm: i16) -> [u32; 4] {
        program([lwl(9, SP, imm), lwr(9, SP, imm), nop(), nop()])
    }

    const fn load_seq_two_imm(a: i16, b: i16) -> [u32; 4] {
        program([lwl(9, SP, a), lwr(9, SP, b), nop(), nop()])
    }

    #[rstest]
    #[case(load_par_program_one_imm(0x0), 0x00ff_ffff, 0x0302_0100)]
    #[case(load_par_program_one_imm(0x1), 0x0100_ffff, 0xff03_0201)]
    #[case(load_par_program_one_imm(0x2), 0x0201_00ff, 0xffff_0302)]
    #[case(load_par_program_one_imm(0x3), 0x0302_0100, 0xffff_ff03)]
    #[case(load_par_program_one_imm(0x4), 0x04ff_ffff, 0x0706_0504)]
    fn test_par_lwl_lwr<const N: usize>(
        #[values(dynarec(), interp())] mut runner: Runner,
        #[case] prog: [u32; N],
        #[case] t1: u32,
        #[case] t2: u32,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
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

        runner.execute(&mut emu);

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
        #[values(dynarec(), interp())] mut runner: Runner,
        #[case] prog: [u32; N],
        #[case] t1: u32,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
        use crate::dynarec_v2::PipelineV2;
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

        runner.execute(&mut emu);

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
        #[values(dynarec(), interp())] mut runner: Runner,
        #[case] prog: [u32; N],
        #[case] word: u32,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
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

        runner.execute(&mut emu);

        tracing::info!("finished running");
        tracing::info!(?emu.cpu);

        let written = emu.read::<u32>(emu.cpu["$sp"]);
        assert_eq_hex!(written, word);

        Ok(())
    }
}

// bios 0x8004f7d0
#[rstest]
fn load_and_compare(#[values(dynarec(), interp())] mut runner: Runner) {
    use crate::cpu::program;
    setup_tracing();

    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.set_reg(10, 420); // poison
    emu.write(0x100, 420);
    emu.write_many(
        0x0,
        &program([
            addiu(9, 0, 69),
            lh(8, 0, 0x100),
            nop(),
            slt(10, 8, 9),
            sw(10, 0, 0x100),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    runner.execute(&mut emu);
    assert_eq!(emu.read::<u32>(0x100), 0)
}
