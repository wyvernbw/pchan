use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use tracing::instrument;

use crate::{Bus, Emu};

// bitfield! {
//     #[derive(Clone, Copy)]
//     pub struct CauseRegister(u32);

//     excode, set_excode: 6, 2;
//     interrupt_pending, set_interrupt_pending: 15, 8;
//     cop_number, set_cop_number: 29, 28;
//     branch_delay, set_branch_delay: 31;
// }

#[bitfield(u32, debug)]
pub struct CauseRegister {
    #[bits(2..=6, rw)]
    excode:      u5,
    #[bit(8, rw)]
    irq_pending: [bool; 8],

    #[bits(8..=15, rw)]
    irq_pending_combined: u8,

    #[bits(28..=29, rw)]
    cop_number: u2,

    #[bit(31)]
    bd: u1,
}

/// # Exception
///
/// enum representing exception code (excode).
/// this enum only holds variants of the currently implemented exceptions.
///
/// Excode Describes what kind of exception occured:
///
/// ```md
/// 00h INT     Interrupt
/// 01h MOD     Tlb modification (none such in PSX)
/// 02h TLBL    Tlb load         (none such in PSX)
/// 03h TLBS    Tlb store        (none such in PSX)
/// 04h AdEL    Address error, Data load or Instruction fetch
/// 05h AdES    Address error, Data store
///             The address errors occur when attempting to read
///             outside of KUseg in user mode and when the address
///             is misaligned. (See also: BadVaddr register)
/// 06h IBE     Bus error on Instruction fetch
/// 07h DBE     Bus error on Data load/store
/// 08h Syscall Generated unconditionally by syscall instruction
/// 09h BP      Breakpoint - break instruction
/// 0Ah RI      Reserved instruction
/// 0Bh CpU     Coprocessor unusable
/// 0Ch Ov      Arithmetic overflow
/// ```
#[bitenum(u5)]
#[derive(Debug)]
pub enum Exception {
    Interrupt = 0x0,
    Syscall   = 0x8,
    Break     = 0x9,
}

pub trait Exceptions: Bus {
    fn handle_exception(&mut self, exception: Exception);
    extern "C" fn handle_rfe(&mut self);
    extern "C" fn handle_break(&mut self);
    extern "C" fn handle_syscall(&mut self);
    fn run_exceptions_io(&mut self);
    fn raise_exception(&mut self, exception: Exception);
    fn clear_exception(&mut self);
}

impl Exceptions for Emu {
    fn handle_exception(&mut self, exception: Exception) {
        let cause = self.cpu().cop0.reg[13];
        let mut cause = CauseRegister::new_with_raw_value(cause);
        cause.set_excode(exception.raw_value());
        self.cpu_mut().cop0.reg[13] = cause.raw_value();

        self.cpu_mut().cop0.reg[14] = self.cpu().pc;

        let sr = self.cpu().cop0.reg[12];
        let new_sr = (sr & !0x3F) | ((sr & 0xF) << 2);
        self.cpu_mut().cop0.reg[12] = new_sr;

        self.cpu_mut().pc = match self.cpu().cop0.status().bev() {
            false => 0x8000_0080,
            true => 0xbfc0_0180,
        }
    }

    #[unsafe(no_mangle)]
    extern "C" fn handle_rfe(&mut self) {
        let sr = self.cpu().cop0.reg[12];
        self.cpu_mut().cop0.reg[12] = (sr & !0x3F) | ((sr >> 2) & 0x3F);
        // panic!("rfe breakpoint");
    }

    #[unsafe(no_mangle)]
    extern "C" fn handle_break(&mut self) {
        self.handle_exception(Exception::Break);
    }

    #[unsafe(no_mangle)]
    extern "C" fn handle_syscall(&mut self) {
        tracing::trace!("syscall");
        self.handle_exception(Exception::Syscall);
    }

    fn run_exceptions_io(&mut self) {
        let sr = self.cpu.cop0.status();
        let cause = self.cpu.cop0.cause();
        // index 2 = bit 10
        if cause.irq_pending(2) && sr.irq_mask(2) && sr.iec() {
            let excode = Exception::new_with_raw_value(cause.excode())
                .expect("unknown exception not yet implemented.");
            self.handle_exception(excode);
        }
    }

    fn raise_exception(&mut self, exception: Exception) {
        self.cpu.cop0.update_cause(|cause| {
            cause
                .with_irq_pending(2, true)
                .with_excode(exception.raw_value())
        });
    }

    fn clear_exception(&mut self) {
        self.cpu
            .cop0
            .update_cause(|cause| cause.with_irq_pending(2, false));
    }
}
