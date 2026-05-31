use crate::Emu;
use crate::cpu::ops::*;
use crate::cpu::program;
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
