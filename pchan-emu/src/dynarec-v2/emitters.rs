use enum_dispatch::enum_dispatch;

#[cfg(test)]
use rstest::rstest;

use crate::Emu;
use crate::cpu::ops::addiu::*;
use crate::cpu::ops::sb::*;
use crate::cpu::ops::sh::*;
use crate::cpu::ops::subu::*;
use crate::cpu::ops::sw::*;
use crate::cpu::ops::{HaltBlock, OpCode};
use crate::dynarec_v2::Dynarec;
use crate::dynarec_v2::Reg;
#[cfg(test)]
use crate::memory;
use crate::memory::ext;
use dynasm::dynasm;
use dynasmrt::DynasmApi;
use dynasmrt::DynasmLabelApi;

#[derive(Default, Debug, Clone, Copy)]
pub enum Boundary {
    #[default]
    None = 0,
    Block,
    Function,
}

#[derive(Debug)]
pub struct EmitCtx<'a> {
    pub dynarec: &'a mut Dynarec,
}

pub struct EmitSummary;

#[enum_dispatch(DecodedOpNew)]
pub trait DynarecOp {
    fn cycles(&self) -> u16 {
        1
    }

    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary;

    fn boundary(&self) -> Boundary {
        Boundary::None
    }

    fn is_block_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Block)
    }

    fn is_function_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Function)
    }

    fn is_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Block | Boundary::Function)
    }
}

#[enum_dispatch]
#[derive(Debug, Clone, Copy, Hash, derive_more::Display)]
pub enum DecodedOpNew {
    ILLEGAL(ILLEGAL),
    ADDIU(ADDIU),
    SUBU(SUBU),
    HaltBlock(HaltBlock),
    SB(SB),
    SH(SH),
    SW(SW),
}

#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
pub struct ILLEGAL;

impl DynarecOp for ILLEGAL {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::None
    }
}

impl DecodedOpNew {
    pub fn new(fields: OpCode) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    pub const fn illegal() -> Self {
        Self::ILLEGAL(ILLEGAL)
    }
    pub fn decode<const N: usize>(fields: [impl Into<OpCode>; N]) -> [Self; N] {
        fields.map(|fields| {
            let fields = fields.into();
            if fields == OpCode::HALT_FIELDS {
                return Self::HaltBlock(HaltBlock);
            }
            let opcode = fields.opcode();
            let rs = fields.rs();
            let rt = fields.rt();
            let rd = fields.rd();
            let funct = fields.funct();
            match (opcode, rs, rt, funct) {
                (0x0, _, _, 0x22 | 0x23) => Self::SUBU(SUBU::new(rd, rs, rt)),
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, fields.imm16())),
                (0x28, _, _, _) => Self::SB(SB::new(rt, rs, fields.imm16())),
                (0x29, _, _, _) => Self::SH(SH::new(rt, rs, fields.imm16())),
                (0x2A, _, _, _) => todo!("swl"),
                (0x2B, _, _, _) => Self::SW(SW::new(rt, rs, fields.imm16())),
                _ => Self::illegal(),
            }
        })
    }
}

impl DynarecOp for ADDIU {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        // $rt = $zero case
        if self.rt == 0 {
            return EmitSummary;
        }

        let rt = ctx.dynarec.alloc_reg(self.rt);

        // $rs = $zero case
        if self.rs == 0 {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov W(*rt), self.imm as _
            );

            rt.restore(ctx.dynarec);
            return EmitSummary;
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; add WSP(*rt), WSP(*rs), self.imm as _
        );

        ctx.dynarec.mark_dirty(self.rt);

        rs.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);

        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::None
    }
}

#[cfg(test)]
#[rstest]
#[case(10, 2, 12)]
fn test_addiu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.write_many(0x0, &program([addiu(12, 10, b as i16), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 1);
    assert_eq!(emu.cpu.pc, 0x4);
    Ok(())
}

impl DynarecOp for HaltBlock {
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::Function
    }
}

impl DynarecOp for SUBU {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        };

        let rd = ctx.dynarec.alloc_reg(self.rd);
        let rs = ctx.dynarec.emit_load_reg(self.rs);
        let rt = ctx.dynarec.emit_load_reg(self.rt);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; sub W(*rd), W(*rs), W(*rt)
        );

        ctx.dynarec.mark_dirty(self.rd);

        rt.restore(ctx.dynarec);
        rs.restore(ctx.dynarec);
        rd.restore(ctx.dynarec);

        EmitSummary
    }
}

#[cfg(test)]
#[rstest]
#[case(10, 2, 8)]
fn test_subu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.cpu.gpr[11] = b;
    emu.write_many(0x0, &program([subu(12, 10, 11), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 1);
    assert_eq!(emu.cpu.pc, 0x4);
    Ok(())
}

fn emit_store<'a, T: 'static>(
    ctx: EmitCtx<'a>,
    rt: u8,
    rs: u8,
    imm: i16,
    func_ptr: unsafe extern "C" fn(*mut Emu, u32, T),
) -> EmitSummary {
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    ctx.dynarec.emit_load_temp_reg(rt, Reg::W(2));

    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; add w1, w1, ext::sign(imm) as _
        ; stp w1, w2, [sp, #-16]!
    );

    ctx.dynarec.set_delay_slot(move |ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr x3, >func_ptr
            ; b >after_ptr
            ; func_ptr:
            ; .u64 func_ptr as usize as _
            ; after_ptr:
            ; ldp w1, w2, [sp], #16
            // we have to store x0 since the function call will clobber it
            ; stp x0, x1, [sp, #-16]!
            // call to function
            ; blr x3
            ; ldp x0, x1, [sp], #16

        );
        EmitSummary
    });

    EmitSummary
}

impl DynarecOp for SB {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm, Emu::write8v2)
    }

    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for SH {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm, Emu::write16v2)
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for SW {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm, Emu::write32v2)
    }
    fn cycles(&self) -> u16 {
        2
    }
}

#[cfg(test)]
#[rstest]
#[case::sb(69u8, sb(10, 11, 2))]
#[case::sh(69u8, sh(10, 11, 2))]
#[case::sw(69u8, sw(10, 11, 2))]
fn test_stores<T>(#[case] value: T, #[case] instr: OpCode) -> color_eyre::Result<()>
where
    T: Copy + std::fmt::Debug + PartialEq,
    T: const memory::Extend<ext::Zero, Out = u32>,
{
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2, memory::ext};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = ext::zero(value);
    emu.cpu.gpr[11] = 0xf;
    emu.write_many(0x0, &program([sb(10, 11, 2), OpCode(69420)]));

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x4);
    assert_eq!(emu.read::<T, ext::Zero>(0xf + 2), ext::zero(value));

    tracing::info!("returning from test...");
    Ok(())
}
