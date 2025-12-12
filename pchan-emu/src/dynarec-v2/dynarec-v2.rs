use bitvec::prelude as bv;
use color_eyre::eyre::bail;
use dynasm::dynasm;
use dynasmrt::Assembler;
use dynasmrt::DynasmApi;
use dynasmrt::ExecutableBuffer;
use pchan_utils::array;

use crate::{cpu::Cpu, memory::Memory};

#[cfg(target_arch = "aarch64")]
type Reloc = dynasmrt::aarch64::Aarch64Relocation;
#[cfg(target_arch = "x86_64")]
type Reloc = dynasmrt::x64::X64Relocation;

pub struct Dynarec {
    reg_alloc: RegAlloc,
    asm: Assembler<Reloc>,
}

impl Default for Dynarec {
    fn default() -> Self {
        let asm = Assembler::new().expect("failed to create assembler");
        Self {
            reg_alloc: Default::default(),
            asm,
        }
    }
}

#[derive(Default)]
pub struct RegAlloc {
    loaded: bv::BitArray,
    dirty: bv::BitArray,
}

pub struct DynarecFunction {
    func: fn(*mut Cpu, *mut Memory),
    exec: ExecutableBuffer,
}

impl Dynarec {
    fn finalize(mut self) -> color_eyre::Result<DynarecFunction> {
        let exec = match self.asm.finalize() {
            Ok(exec) => exec,
            Err(asm) => {
                self.asm = asm;
                bail!("failed to assemble function");
            }
        };
        let func = unsafe { std::mem::transmute::<*const u8, fn(_, _)>(exec.as_ptr()) };
        Ok(DynarecFunction { func, exec })
    }
    fn emit_block_prelude(&mut self, cpu: *const Cpu, mem: *const Memory) {
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(
                self.asm
                ; .arch aarch64
                // TODO: save all registers
                ; stp x23, x24, [sp, -16]!
                ; stp x25, x26, [sp, -16]!
                ; stp x27, x28, [sp, -16]!
            )
        }
    }
    fn emit_writeback_free(asm: &mut Assembler<Reloc>, guest_reg: u8, host_reg: Reg) {
        let offset = u32::try_from(Cpu::reg_offset(guest_reg)).expect("invalid cpu offset");

        // emit writeback
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            let Reg::W(host_reg) = host_reg;

            dynasm!(
                asm
                ; .arch aarch64
                ; str W(host_reg), [x0, offset]
            )
        }
    }

    #[inline(always)]
    fn emit_writeback(&mut self, guest_reg: u8, host_reg: Reg) {
        Self::emit_writeback_free(&mut self.asm, guest_reg, host_reg);
    }

    fn emit_block_epilogue(&mut self) {
        // emit write back to dirty registers
        self.reg_alloc
            .dirty
            .iter()
            .enumerate()
            .flat_map(|(guest_reg, dirty)| if *dirty { Some(guest_reg) } else { None })
            .flat_map(|guest_reg| {
                let host_reg = REG_MAP[guest_reg];
                Some(guest_reg as u8).zip(host_reg)
            })
            .for_each(|(guest_reg, host_reg)| {
                Self::emit_writeback_free(&mut self.asm, guest_reg, host_reg);
            });

        // emit return
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            dynasm!(
                self.asm
                ; .arch aarch64
                // TODO: restore all registers
                ; ldp x27, x28, [sp], #16
                ; ldp x25, x26, [sp], #16
                ; ldp x23, x24, [sp], #16
                ; ret
            )
        }
    }
    fn emit_load_reg(&mut self, guest_reg: u8, spill_to: Reg) -> Result<Reg, Reg> {
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            let host_reg = REG_MAP[guest_reg as usize];
            match host_reg {
                Some(Reg::W(host_reg)) => {
                    // already in register
                    if self.reg_alloc.loaded[guest_reg as usize] {
                        return Ok(Reg::W(host_reg));
                    }

                    let offset =
                        u32::try_from(Cpu::reg_offset(guest_reg)).expect("invalid cpu offset");

                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; ldr W(host_reg), [x0, offset]
                    );

                    self.reg_alloc.loaded.set(guest_reg as usize, true);
                    self.reg_alloc.dirty.set(guest_reg as usize, false);

                    Ok(Reg::W(host_reg))
                }
                // spill register
                None => {
                    let Reg::W(spill_to) = spill_to;
                    let offset =
                        u32::try_from(Cpu::reg_offset(guest_reg)).expect("invalid cpu offset");
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; ldr W(spill_to), [x0, offset]
                    );
                    self.reg_alloc.dirty.set(guest_reg as usize, false);

                    Err(Reg::W(spill_to))
                }
            }
        }
    }
    fn mark_dirty(&mut self, guest_reg: u8) {
        self.reg_alloc.dirty.set(guest_reg as usize, true);
    }
    fn mark_clean(&mut self, guest_reg: u8) {
        self.reg_alloc.dirty.set(guest_reg as usize, false);
    }
}

pub trait ResultIntoInner {
    type IntoInner;
    fn into_inner(self) -> Self::IntoInner;
}

impl<T> ResultIntoInner for Result<T, T> {
    type IntoInner = T;

    fn into_inner(self) -> Self::IntoInner {
        match self {
            Ok(ok) => ok,
            Err(err) => err,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Boundary {
    #[default]
    None = 0,
    Block,
    Function,
}

pub struct EmitCtx<'a> {
    dynarec: &'a mut Dynarec,
}

pub struct EmitSummary;

pub trait DynarecOp {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary;

    fn boundary(&self) -> Boundary;

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

#[derive(Debug, Clone, Copy)]
enum Reg {
    #[cfg(target_arch = "aarch64")]
    W(u8),
}

#[cfg(target_arch = "aarch64")]
impl Reg {
    const WZR: Self = Reg::W(31);
    const WSP: Self = Reg::W(31);
}

#[cfg(target_arch = "aarch64")]
impl From<Reg> for u8 {
    fn from(value: Reg) -> Self {
        match value {
            Reg::W(reg) => reg,
        }
    }
}

#[cfg(target_arch = "aarch64")]
static REG_MAP: [Option<Reg>; 32] = array![
    // $zero
    0 => Some(Reg::WZR),

    // $at
    1 => Some(Reg::W(1)),

    // $v0, $v1
    2 => Some(Reg::W(2)),
    3 => Some(Reg::W(3)),

    // $a0-$a3
    4 => Some(Reg::W(4)),
    5 => Some(Reg::W(5)),
    6 => Some(Reg::W(6)),
    7 => Some(Reg::W(7)),

    // $t0-$t7
    8 => Some(Reg::W(8)),
    9 => Some(Reg::W(9)),
    10 => Some(Reg::W(10)),
    11 => Some(Reg::W(11)),
    12 => Some(Reg::W(12)),
    13 => Some(Reg::W(13)),
    14 => Some(Reg::W(14)),
    15 => Some(Reg::W(15)),

    // $s0-$s7
    16 => Some(Reg::W(16)),
    17 => Some(Reg::W(17)),
    18 => Some(Reg::W(18)),
    19 => Some(Reg::W(19)),
    20 => Some(Reg::W(20)),
    21 => Some(Reg::W(21)),
    22 => Some(Reg::W(22)),
    23 => Some(Reg::W(23)),

    // $t8-$t9
    24 => None,
    25 => None,

    // $k0-$k1
    26 => None,
    27 => None,

    // $gp
    28 => None,

    // $sp
    29 => Some(Reg::WSP),

    // $fp
    30 => None,

    // $ra
    31 => Some(Reg::W(24))
];

#[cfg(test)]
mod tests {
    use std::ptr;

    use color_eyre::Result;
    use pchan_utils::setup_tracing;

    use super::*;
    use crate::{Emu, cpu::ops::addiu::*};

    impl DynarecOp for ADDIU {
        fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
            use dynasm::dynasm;
            #[cfg(target_arch = "aarch64")]
            {
                let rs = ctx.dynarec.emit_load_reg(self.rs, Reg::W(25)).into_inner();
                let loaded_rt = ctx.dynarec.emit_load_reg(self.rt, Reg::W(26));
                let rt = loaded_rt.into_inner();
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; add WSP(rt), WSP(rs), self.imm as _
                );

                if loaded_rt.is_ok() {
                    ctx.dynarec.mark_dirty(self.rt);
                } else {
                    ctx.dynarec.emit_writeback(self.rt, rt);
                }
            }
            EmitSummary
        }

        fn boundary(&self) -> Boundary {
            Boundary::None
        }
    }

    #[test]
    fn dynarec_minimal_test() -> Result<()> {
        setup_tracing();
        let mut emu = Emu::default();
        let mut dynarec = Dynarec::default();

        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(dynarec.asm; .arch aarch64; ret);
        }

        let func = dynarec.finalize()?;
        tracing::info!("Calling JIT function...");
        func.func.call((&mut emu.cpu, &mut emu.mem));
        tracing::info!("JIT call succeeded!");

        Ok(())
    }

    #[test]
    fn dynarec_v2_test() -> Result<()> {
        setup_tracing();
        let mut emu = Emu::default();
        let mut dynarec = Dynarec::default();

        dynarec.emit_block_prelude(ptr::from_ref(&emu.cpu), ptr::from_ref(&emu.mem));
        tracing::info!("emitted prelude");
        ADDIU::new(9, 8, 69).emit(EmitCtx {
            dynarec: &mut dynarec,
        });
        tracing::info!("emitted addiu instruction");
        dynarec.emit_block_epilogue();
        tracing::info!("emitted epilogue");

        let func = dynarec.finalize()?;
        func.func.call((&mut emu.cpu, &mut emu.mem));

        tracing::info!(?emu.cpu);
        assert_eq!(emu.cpu.gpr[8], 69);

        Ok(())
    }

    #[test]
    fn dynarec_v2_test_register_spill() -> Result<()> {
        setup_tracing();
        let mut emu = Emu::default();
        let mut dynarec = Dynarec::default();

        dynarec.emit_block_prelude(ptr::from_ref(&emu.cpu), ptr::from_ref(&emu.mem));
        tracing::info!("emitted prelude");
        ADDIU::new(9, 24, 69).emit(EmitCtx {
            dynarec: &mut dynarec,
        });
        tracing::info!("emitted addiu instruction");
        dynarec.emit_block_epilogue();
        tracing::info!("emitted epilogue");
        let func = dynarec.finalize()?;
        tracing::info!("About to call JIT function");
        func.func.call((&mut emu.cpu, &mut emu.mem));
        tracing::info!("JIT function returned successfully");
        tracing::info!(?emu.cpu);
        tracing::info!("About to assert");
        assert_eq!(emu.cpu.gpr[24], 69);
        tracing::info!("Assert passed");

        Ok(())
    }
}
