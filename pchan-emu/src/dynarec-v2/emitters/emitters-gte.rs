use crate::cpu::Cpu;
use crate::cpu::gte::{Gte, GteBindings};
use crate::dynarec_v2::emitters::{EmitCtx, EmitSummary, emit_call, shadow};
use crate::dynarec_v2::regalloc::Reg;
use core::alloc::Allocator;
use dynasm::dynasm;
use dynasmrt::DynasmApi;

#[allow(clippy::useless_conversion)]
pub(crate) fn emit_mtc2_ctc2<A: Allocator + Copy>(mut ctx: EmitCtx<A>, rt: u8, rd: u8) {
    match rd {
        // trunacte u32 to i16 then sign extend
        1 | 3 | 5 | 8 | 9 | 10 | 11 | 36 | 44 | 52 | 58 | 59 | 61 | 62 => {
            ctx.dynarec.emit_load_temp_reg(rt, Reg::W(1));
            dynasm!(
                ctx.dynarec.asm
                ; sxth w1, w1
                ; str w1, [x0, #Cpu::cop_reg_offset(2, rd) as u32]
                ; .arch aarch64
                ; nop
            )
        }
        // sz fifo
        7 | 16..=19 => {
            ctx.dynarec.emit_load_temp_reg(rt, Reg::W(1));
            dynasm!(
                ctx.dynarec.asm
                ; uxth w1, w1
                ; str w1, [x0, #Cpu::cop_reg_offset(2, rd) as u32]
                ; .arch aarch64
                ; nop
            )
        }
        // ORGB, LZCR: read only
        29 | 31 => {}
        // gte screen z fifo mirror register "SXYP"
        15 => {
            emit_call(&mut ctx, |dynarec| {
                dynarec.emit_load_temp_reg(rt, Reg::W(1));
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_SXYP]
                    ; blr x3
                );
            });
        }
        // gte irgb register
        28 => {
            let rt = ctx.dynarec.emit_load_reg_stackless(rt);
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; mov w1, W(*rt)
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_IRGB]
                    ; blr x3
                );
            });
        }
        // gte lzcs register
        30 => {
            let rt = ctx.dynarec.emit_load_reg_stackless(rt);
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; mov w1, W(*rt)
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_LZCS]
                    ; blr x3
                );
            });
        }
        // gte flag
        63 => {
            emit_call(&mut ctx, |dynarec| {
                dynarec.emit_load_temp_reg(rt, Reg::W(1));
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_FLAG]
                    ; blr x3
                );
            });
        }
        _ => {
            let rt = ctx.dynarec.emit_load_reg_stackless(rt);

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; nop
                ; str W(*rt), [x0, Cpu::cop_reg_offset(2, rd) as _]
                ; nop
            );
        }
    }
}

#[allow(clippy::useless_conversion)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn emit_mfc2_cfc2<A: Allocator + Copy>(mut ctx: EmitCtx<A>, rt: u8, rd: u8) {
    let rta = ctx.dynarec.alloc_reg_stackless(rt);

    match rd {
        // irgb/orgb, call read_irgb_orgb to update value from irv, then read
        28 | 29 => {
            emit_call(&mut ctx, |dynarec| {
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; ldr x3, [x0, #GteBindings::<A>::READ_IRGB_ORGB]
                    ; blr x3
                );
            });
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, Gte::RGB_U32]
            );
        }
        // H: buggy u16, sign extend on read
        58 => {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, Cpu::cop_reg_offset(2, rd) as _]
                ; sxth W(*rta), W(*rta)
            );
        }
        // FLAG: u20, zero out bottom 12 bits
        63 => {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, Cpu::cop_reg_offset(2, rd) as _]
                ; bfc W(*rta), 0, 12
            );
        }
        _ => {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, Cpu::cop_reg_offset(2, rd) as _]
            );
        }
    }

    if rt != 0 {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; str W(*rta), [x0, #shadow(rt)]
        );
    }

    // writeback to gpr
    ctx.schedule_in(1, Some(rt), move |ctx| {
        let rta = ctx.dynarec.alloc_reg_stackless(rt);
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rta), [x0, #shadow(rt)]
        );
        ctx.dynarec.mark_dirty(rt);
        EmitSummary::default()
    });
}
