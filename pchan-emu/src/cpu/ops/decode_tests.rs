use pchan_utils::hex;
use pchan_utils::setup_tracing;
use pretty_assertions::assert_eq;
use rstest::rstest;

use crate::Emu;
use crate::cpu::ops::DecodedOp;
use crate::dynarec::prelude::*;

#[rstest]
fn test_bios_ops(setup_tracing: ()) -> color_eyre::Result<()> {
    let mut emu = Emu::default();
    emu.load_bios()?;

    let ops = (0xbfc0_0000u32..0xbfc0_0000u32 + 32 * 4)
        .step_by(4)
        .map(|address| (address, emu.read::<OpCode, ext::NoExt>(address)))
        .map(|(address, op)| (address, DecodedOp::new(op)))
        .inspect(|(address, op)| tracing::info!("{}: {}", hex(*address), op))
        .collect::<Vec<_>>();

    Ok(())
}

#[rstest]
#[case::nop(DecodedOp::new(nop()), "nop")]
#[case::lb(DecodedOp::new(lb(8, 9, 4)), "lb $t0 $t1 0x0004")]
#[case::lbu(DecodedOp::new(lbu(8, 9, 4)), "lbu $t0 $t1 0x0004")]
#[case::lh(DecodedOp::new(lh(8, 9, 4)), "lh $t0 $t1 0x0004")]
#[case::lhu(DecodedOp::new(lhu(8, 9, 4)), "lhu $t0 $t1 0x0004")]
#[case::lw(DecodedOp::new(lw(8, 9, 4)), "lw $t0 $t1 0x0004")]
#[case::sb(DecodedOp::new(sb(8, 9, 4)), "sb $t0 $t1 0x0004")]
#[case::sh(DecodedOp::new(sh(8, 9, 4)), "sh $t0 $t1 0x0004")]
#[case::sw(DecodedOp::new(sw(8, 9, 4)), "sw $t0 $t1 0x0004")]
#[case::addu(DecodedOp::new(addu(8, 9, 10)), "addu $t0 $t1 $t2")]
#[case::addiu(DecodedOp::new(addiu(8, 9, 123)), "addiu $t0 $t1 0x007b")]
#[case::subu(DecodedOp::new(subu(8, 9, 10)), "subu $t0 $t1 $t2")]
#[case::j(DecodedOp::new(j(0x0000_2000 >> 2)), "j 0x00002000")]
#[case::beq(DecodedOp::new(beq(8, 9, 16)), "beq $t0 $t1 0x0044")]
#[case::bne(DecodedOp::new(bne(8, 9, 16)), "bne $t0 $t1 0x0044")]
#[case::slt(DecodedOp::new(slt(8, 9, 10)), "slt $t0 $t1 $t2")]
#[case::sltu(DecodedOp::new(sltu(8, 9, 10)), "sltu $t0 $t1 $t2")]
#[case::slti(DecodedOp::new(slti(8, 9, 32)), "slti $t0 $t1 0x0020")]
#[case::sltiu(DecodedOp::new(sltiu(8, 9, 32)), "sltiu $t0 $t1 0x0020")]
#[case::and(DecodedOp::new(and(8, 9, 10)), "and $t0 $t1 $t2")]
#[case::or(DecodedOp::new(or(8, 9, 10)), "or $t0 $t1 $t2")]
#[case::xor(DecodedOp::new(xor(8, 9, 10)), "xor $t0 $t1 $t2")]
#[case::nor(DecodedOp::new(nor(8, 9, 10)), "nor $t0 $t1 $t2")]
#[case::andi(DecodedOp::new(andi(8, 9, 4)), "andi $t0 $t1 0x0004")]
#[case::ori(DecodedOp::new(ori(8, 9, 4)), "ori $t0 $t1 0x0004")]
#[case::xori(DecodedOp::new(xori(8, 9, 4)), "xori $t0 $t1 0x0004")]
#[case::sllv(DecodedOp::new(sllv(8, 9, 10)), "sllv $t0 $t1 $t2")]
#[case::srlv(DecodedOp::new(srlv(8, 9, 10)), "srlv $t0 $t1 $t2")]
#[case::srav(DecodedOp::new(srav(8, 9, 10)), "srav $t0 $t1 $t2")]
#[case::sll(DecodedOp::new(sll(8, 9, 4)), "sll $t0 $t1 0x04")]
#[case::srl(DecodedOp::new(srl(8, 9, 4)), "srl $t0 $t1 0x04")]
#[case::sra(DecodedOp::new(sra(8, 9, 4)), "sra $t0 $t1 0x04")]
#[case::lui(DecodedOp::new(lui(8, 32)), "lui $t0 0x0020")]
#[case::mult(DecodedOp::new(mult(8, 9)), "mult $t0 $t1")]
#[case::jal(DecodedOp::new(jal(0x0040_0000 >> 2)), "jal 0x00400000")]
#[case::multu(DecodedOp::new(multu(8, 9)), "multu $t0 $t1")]
#[case::mflo(DecodedOp::new(mflo(8)), "mflo $t0")]
#[case::mfhi(DecodedOp::new(mfhi(8)), "mfhi $t0")]
#[case::mthi(DecodedOp::new(mthi(8)), "mthi $t0")]
#[case::mtlo(DecodedOp::new(mtlo(8)), "mtlo $t0")]
#[case::jr(DecodedOp::new(jr(8)), "jr $t0")]
#[case::jalr(DecodedOp::new(jalr(8, 9)), "jalr $t0 $t1")]
#[case::mtc(DecodedOp::new(mtc0(8, 16)), "mtc0 $t0, $r16")]
#[case::mfc(DecodedOp::new(mfc0(8, 16)), "mfc0 $t0, $r16")]
#[case::rfe(DecodedOp::new(rfe()), "rfe")]
#[case::bgez(DecodedOp::new(bgez(8, 0x20)), "bgez $t0 0x0020")]
#[case::addiu_02(DecodedOp::new(addiu(26, 26, 0x0C80)), "addiu $k0 $k0 0x0c80")]
fn test_display(setup_tracing: (), #[case] op: DecodedOp, #[case] expected: &str) {
    assert_eq!(op.to_string(), expected);
}

#[rstest]
#[case::addiu(0x275a0c80, "addiu $k0 $k0 0x0c80")]
fn test_decode(setup_tracing: (), #[case] opcode: u32, #[case] expected: &str) {
    assert_eq!(
        DecodedOp::new(OpCode(opcode)).to_string(),
        "addiu $k0 $k0 0x0c80"
    );
}
