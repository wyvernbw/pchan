use alloc::sync::Arc;
use core::alloc::Allocator;
use core::cell::Cell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};
use core::{cmp, iter, mem};
use derive_more as d;
use dynasm::dynasm;
use dynasmrt::{Assembler, DynasmApi, DynasmLabelApi, ExecutableBuffer};
use heapless::binary_heap::Min;
use pchan_utils::{default, hex, max_simd_elements};
use smallbox::SmallBox;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::simd::Simd;
use thiserror::Error;
use tracing::{Instrument, Level, enabled};

use crate::cpu::exceptions::Exception;
use crate::cpu::ops::OpCode;
use crate::cpu::reg_str;
use crate::dynarec_v2::emitters::{DecodedOp, DynarecOp, EmitCtx, EmitSummary};
use crate::dynarec_v2::regalloc::{
    AllocResult, AllocResultStackless, Guest, Reg, RegAlloc, RegAllocError, RegAllocErrorStackless,
    RegisterType,
};
use crate::memory::kb;
use crate::{AllocatorClone, Emu};

pub mod emitters;
pub mod regalloc;

pub static INSTR_COMPILED: AtomicU64 = AtomicU64::new(0);
pub static INSTR_EXECUTED: AtomicU64 = AtomicU64::new(0);
pub static BLOCKS_COMPILED: AtomicU64 = AtomicU64::new(0);
pub static BLOCKS_EXECUTED: AtomicU64 = AtomicU64::new(0);
pub static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

static ASM_CAPACITY: usize = 64 * size_of::<usize>();

pub fn cache_hitrate() -> f32 {
    let hits = CACHE_HITS.load(Ordering::Relaxed) as f32;
    let misses = CACHE_MISSES.load(Ordering::Relaxed) as f32;
    hits / (hits + misses)
}

#[cfg(feature = "fetch-channel")]
pub mod fetch_map {
    use alloc::sync::Arc;
    use std::collections::HashMap;
    use std::sync::{LazyLock, RwLock};

    use crate::dynarec_v2::emitters::DecodedOp;

    pub static FETCH_MAP: LazyLock<RwLock<HashMap<u32, Arc<[DecodedOp]>>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));

    pub fn fetch_map_insert(pc: u32, ops: impl Into<Arc<[DecodedOp]>>) {
        FETCH_MAP.write().unwrap().insert(pc, ops.into());
    }

    pub fn fetch_map_get(pc: u32) -> Option<Arc<[DecodedOp]>> {
        FETCH_MAP.read().unwrap().get(&pc).cloned()
    }
}

#[cfg(target_arch = "aarch64")]
type Reloc = dynasmrt::aarch64::Aarch64Relocation;
#[cfg(target_arch = "x86_64")]
type Reloc = dynasmrt::x64::X64Relocation;

type DynEmitter<A> = SmallBox<dyn for<'a> Fn(EmitCtx<'a, A>) -> EmitSummary, [usize; 1]>;

#[derive(derive_more::Debug)]
pub struct Dynarec<A: Allocator + Copy> {
    pub(crate) reg_alloc:  RegAlloc,
    pub(crate) scheduler:  Box<Scheduler<A>, A>,
    pub last_ran_function: Option<DynarecFunction<A>>,
    #[debug(skip)]
    asm:                   Assembler<Reloc>,
}

pub struct CreateDynarecParams<A: Allocator + Copy> {
    reg_alloc: Option<RegAlloc>,
    scheduler: Option<Box<Scheduler<A>, A>>,
    asm:       Option<Assembler<Reloc>>,
    alloc:     A,
}

impl<A: Allocator + Copy> CreateDynarecParams<A> {
    pub fn new(alloc: A) -> Self {
        Self {
            reg_alloc: None,
            scheduler: None,
            asm: None,
            alloc,
        }
    }
}

unsafe impl<A: Allocator + Copy> Send for Dynarec<A> {}
unsafe impl<A: Allocator + Copy> Sync for Dynarec<A> {}

impl<A: Allocator + Copy> Dynarec<A> {
    pub fn new(
        CreateDynarecParams {
            reg_alloc,
            scheduler,
            asm,
            alloc,
        }: CreateDynarecParams<A>,
    ) -> Self {
        let reg_alloc = reg_alloc.unwrap_or_default();
        let scheduler = scheduler.unwrap_or_else(|| Box::new_in(Scheduler::<A>::default(), alloc));
        let asm = asm.unwrap_or_else(|| {
            Assembler::new_with_capacity(ASM_CAPACITY).expect("fatal: failed to allocate assembler")
        });
        Self {
            reg_alloc,
            scheduler,
            asm,
            last_ran_function: None,
        }
    }

    pub fn reset(&mut self, alloc: A) {
        self.reg_alloc = default();
        self.scheduler = Box::new_in(Scheduler::default(), alloc);
        self.asm =
            Assembler::new_with_capacity(ASM_CAPACITY).expect("fatal failed to allocate assembler");
    }
}

#[derive(Debug, Clone)]
pub struct DynarecFunction<A: Allocator + Copy> {
    pub func: fn(*mut Emu<A>),
    pub exec: Arc<ExecutableBuffer>,
}

#[derive(Debug, Clone)]
pub struct DynarecBlock<A: Allocator + Copy> {
    pub(crate) function: DynarecFunction<A>,
    pub(crate) pc:       u32,
    pub(crate) op_count: u32,
}

type DynarecBlockArgs<'a, A: Allocator + Copy> = (&'a mut Emu<A>, bool);

impl<A: Allocator + Copy> DynarecBlock<A> {
    pub fn call_block(&self, (emu, instrument): DynarecBlockArgs<A>) {
        #[cfg(debug_assertions)]
        {
            BLOCKS_EXECUTED.fetch_add(1, Ordering::Relaxed);
            INSTR_EXECUTED.fetch_add(self.op_count as u64, Ordering::Relaxed);
        }

        // reset delta clock before running
        emu.cpu.d_clock = 0;

        if instrument {
            (self.function.func)
                .instrument(tracing::info_span!("fn", addr = ?self.function.func))
                .inner()(emu)
        } else {
            (self.function.func)(emu)
        };

        #[cfg(feature = "debugger-ext")]
        {
            use crate::debug::BreakpointKind;

            emu.dbg.break_on(emu.cpu.pc, BreakpointKind::EXECUTE);
        }

        emu.cpu.drain_jump_queue();
        if emu.cpu.pc % 0x4 != 0 {
            emu.raise_exception(Exception::AdEl);
        }
        emu.run_io();
        emu.cpu.drain_jump_queue();
        if emu.cpu.pc % 0x4 != 0 {
            emu.raise_exception(Exception::AdEl);
        }
        emu.cpu.cop0.set_bd(false);

        debug_assert_eq!(emu.cpu.gpr[0], 0);
    }

    pub fn buffer(&self) -> &ExecutableBuffer {
        &self.function.exec
    }
}

impl<A: Allocator + Copy> FnMut<DynarecBlockArgs<'_, A>> for DynarecBlock<A> {
    extern "rust-call" fn call_mut(&mut self, args: DynarecBlockArgs<A>) -> Self::Output {
        self.call_block(args)
    }
}

impl<A: Allocator + Copy> FnOnce<DynarecBlockArgs<'_, A>> for DynarecBlock<A> {
    type Output = ();
    extern "rust-call" fn call_once(mut self, args: DynarecBlockArgs<A>) -> Self::Output {
        self.call_mut(args)
    }
}

impl<A: Allocator + Copy> Fn<DynarecBlockArgs<'_, A>> for DynarecBlock<A> {
    extern "rust-call" fn call(&self, args: DynarecBlockArgs<A>) -> Self::Output {
        self.call_block(args);
    }
}

#[derive(Error, Debug)]
pub(crate) enum FinalizeError {
    #[error("failed to assemble")]
    AssembleError,
    #[error("dynarec: io error {0}")]
    IoError(#[from] std::io::Error),
}

impl<A: Allocator + Copy> Dynarec<A> {
    pub(crate) fn finalize(&mut self) -> Result<DynarecFunction<A>, FinalizeError> {
        self.scheduler.queue.clear();
        self.reg_alloc = RegAlloc::default();
        let asm = mem::replace(
            &mut self.asm,
            Assembler::new_with_capacity(ASM_CAPACITY).unwrap(),
        );
        let exec = match asm.finalize() {
            Ok(exec) => exec,
            Err(asm) => {
                self.asm = asm;
                return Err(FinalizeError::AssembleError);
            }
        };

        if enabled!(Level::DEBUG) {
            use std::fs::File;
            use std::io::Write;
            File::create("/tmp/jit_code.bin")?.write_all(exec.as_ref())?;
            tracing::trace!("Wrote {} bytes to /tmp/jit_code.bin", exec.len());
        }

        let func = unsafe { mem::transmute::<*const u8, fn(*mut Emu<A>)>(exec.as_ptr()) };
        Ok(DynarecFunction {
            func,
            exec: Arc::new(exec),
        })
    }
    pub(crate) fn emit_block_prelude(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(
                self.asm
                ; .arch aarch64
                ; b >after_table

                // -- function table --
                ; -> write32v2:
                ; .u64 Emu::<A>::write32v2 as *const () as _
                ; -> write16v2:
                ; .u64 Emu::<A>::write16v2 as *const () as _
                ; -> write8v2:
                ; .u64 Emu::<A>::write8v2 as *const () as _
                ; -> ulwrite32:
                ; .u64 Emu::<A>::ulwrite32 as *const () as _
                ; -> urwrite32:
                ; .u64 Emu::<A>::urwrite32 as *const () as _
                ; -> readi8v2:
                ; .u64 Emu::<A>::readi8v2 as *const () as _
                ; -> readu8v2:
                ; .u64 Emu::<A>::readu8v2 as *const () as _
                ; -> readi16v2:
                ; .u64 Emu::<A>::readi16v2 as *const () as _
                ; -> readu16v2:
                ; .u64 Emu::<A>::readu16v2 as *const () as _
                ; -> read32v2:
                ; .u64 Emu::<A>::read32v2 as *const () as _
                ; -> ulread32:
                ; .u64 Emu::<A>::ulread32 as *const () as _
                ; -> urread32:
                ; .u64 Emu::<A>::urread32 as *const () as _
                ; -> handle_syscall:
                ; .u64 Emu::<A>::handle_syscall as *const () as _
                ; -> handle_rfe:
                ; .u64 Emu::<A>::handle_rfe as *const () as _
                ; -> jump:
                ; .u64 DynarecCache::<A>::jump as *const () as _
                ; -> run_io:
                ; .u64 Emu::<A>::ext_run_io as *const () as _
                ; after_table:

                ; stp x19, x20, [sp, -16]!
                ; stp x21, x22, [sp, -16]!
                ; stp x23, x24, [sp, -16]!
                ; stp x25, x26, [sp, -16]!
                ; stp x27, x28, [sp, -16]!
                ; stp x29, x30, [sp, -16]!
            )
        }
    }
    fn emit_writeback_free(asm: &mut Assembler<Reloc>, guest_reg: u8, host_reg: Reg) {
        let offset = Emu::<A>::reg_offset(guest_reg) as u32;

        if enabled!(Level::TRACE) {
            tracing::trace!("store: guest r{}", guest_reg);
        }

        // emit writeback
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            let Reg::W(host_reg) = host_reg;
            assert!(host_reg != 0, "cannot writeback to zero register");

            dynasm!(
                asm
                ; .arch aarch64
                ; str W(host_reg), [x0, offset]
            )
        }
    }

    fn emit_writeback_pair_free(asm: &mut Assembler<Reloc>, arr: [(u8, Reg); 2]) {
        let [(gr1, hr1), (gr2, hr2)] = arr;
        debug_assert!(
            hr1.consecutive(hr2),
            "pairs store must be of consecutive registers"
        );
        assert!(gr1 != 0 && gr2 != 0, "cannot writeback to zero register");

        let offset = Emu::<A>::reg_offset(gr1) as u32;

        if enabled!(Level::TRACE) {
            tracing::trace!("store: guest ${} & ${} (pair)", reg_str(gr1), reg_str(gr2));
        }

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            asm
            ; .arch aarch64
            ; stp W(hr1), W(hr2), [x0, offset as _]
        );
    }

    #[inline(always)]
    fn emit_writeback(&mut self, guest_reg: u8, host_reg: Reg) {
        Self::emit_writeback_free(&mut self.asm, guest_reg, host_reg);
    }

    fn emit_writeback_all(&mut self) {
        // emit write back to dirty registers
        self.reg_alloc
            .dirty
            .clone() // this is actually cheap since `dirty` is just a u32
            .iter()
            .enumerate()
            .flat_map(|(guest_reg, dirty)| if *dirty { Some(guest_reg) } else { None })
            .for_each(|guest_reg| {
                let host_reg = self.alloc_reg(guest_reg as _);
                self.emit_writeback(guest_reg as _, host_reg.reg());
            });
    }

    pub(crate) fn emit_block_epilogue(
        &mut self,
        d_clock: u32,
        new_pc: Option<u32>,
        emit_ret: bool,
    ) {
        self.emit_writeback_all();

        // emit pc update & clock update
        match new_pc {
            Some(new_pc) => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; .arch aarch64
                    ; movz w24, new_pc >> 16 , LSL #16
                    ; movk w24, new_pc & 0x0000_FFFF
                    ; movz w25, d_clock >> 16 , LSL #16
                    ; movk w25, d_clock & 0x0000_FFFF
                    ; stp w24, w25, [x0, Emu::<A>::PC_OFFSET as _]
                )
            }
            None => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; .arch aarch64
                    ; movz w25, d_clock >> 16 , LSL #16
                    ; movk w25, d_clock & 0x0000_FFFF
                    ; str w25, [x0, Emu::<A>::D_CLOCK_OFFSET as _]
                )
            }
        };

        // emit return
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            dynasm!(
                self.asm
                ; .arch aarch64
                ; ldp x29, x30, [sp], #16
                ; ldp x27, x28, [sp], #16
                ; ldp x25, x26, [sp], #16
                ; ldp x23, x24, [sp], #16
                ; ldp x21, x22, [sp], #16
                ; ldp x19, x20, [sp], #16
            )
        }
        if emit_ret {
            #[cfg(target_arch = "aarch64")]
            #[allow(clippy::useless_conversion)]
            {
                dynasm!(
                    self.asm
                    ; .arch aarch64
                    ; ret
                )
            }
        }
    }
    fn alloc_reg_stackless(&mut self, guest_reg: u8) -> LoadedReg<AllocResultStackless> {
        let result = self.reg_alloc.regalloc_stackless(guest_reg);
        match result {
            // no op case
            Err(RegAllocErrorStackless::AlreadyAllocatedTo(_)) => {}
            // new allocation
            Ok(_) => {
                self.reg_alloc.dirty.set(guest_reg as usize, false);
            }
            // spill register
            Err(RegAllocErrorStackless::EvictToMemory(evicted_guest, host_reg)) => {
                self.emit_writeback(*evicted_guest, *host_reg);
                self.reg_alloc.dirty.set(*evicted_guest as usize, false);
                self.reg_alloc.dirty.set(guest_reg as usize, false);
            }
        };
        LoadedReg::from(result)
    }
    fn alloc_reg(&mut self, guest_reg: u8) -> LoadedReg<AllocResult> {
        let result = self.reg_alloc.regalloc(guest_reg);
        match result {
            // no op case
            Err(RegAllocError::AlreadyAllocatedTo(_)) => {}
            // new allocation
            Ok(_) => {
                self.reg_alloc.dirty.set(guest_reg as usize, false);
            }
            // spill register
            Err(RegAllocError::EvictToMemory(evicted_guest, host_reg)) => {
                self.emit_writeback(*evicted_guest, *host_reg);
                self.reg_alloc.dirty.set(*evicted_guest as usize, false);
                self.reg_alloc.dirty.set(guest_reg as usize, false);
            }
            Err(RegAllocError::EvictToStack(_, host_reg)) => {
                dynasm!(
                    self.asm
                    ; .arch aarch64
                    ; str W(host_reg), [sp, #-16]!
                );
                self.reg_alloc.dirty.set(guest_reg as usize, false);
            }
        };
        LoadedReg::from(result)
    }
    fn emit_load_temp_reg(&mut self, guest_reg: u8, host_reg: Reg) {
        debug_assert!(!self.reg_alloc.allocatable[host_reg.to_idx() as usize]);

        if enabled!(Level::TRACE) {
            tracing::trace!("load: guest r{} to temp reg {:?}", guest_reg, host_reg);
        }

        if let Some(allocated) = self.reg_alloc.mapping[guest_reg as usize] {
            dynasm!(
                self.asm
                ; .arch aarch64
                ; mov W(host_reg), W(allocated)
            );
            return;
        }

        if guest_reg == 0 {
            dynasm!(
                self.asm
                ; .arch aarch64
                ; mov W(host_reg), 0
            );
        } else {
            let offset = Emu::<A>::reg_offset(guest_reg) as u32;
            dynasm!(
                self.asm
                ; .arch aarch64
                ; ldr W(host_reg), [x0, offset]
            );
        }
    }
    fn emit_load_reg(&mut self, guest_reg: u8) -> LoadedReg<AllocResult> {
        let host_reg = self.alloc_reg(guest_reg);
        match host_reg.result {
            Err(RegAllocError::AlreadyAllocatedTo(_)) => {}
            _ => self.impl_emit_reg_load(guest_reg, &host_reg),
        }
        host_reg
    }

    fn impl_emit_reg_load<T>(&mut self, guest_reg: u8, host_reg: &LoadedReg<T>) {
        let offset = Emu::<A>::reg_offset(guest_reg) as u32;
        match guest_reg == 0 {
            true => {
                #[cfg(target_arch = "aarch64")]
                #[allow(clippy::useless_conversion)]
                {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; mov W(**host_reg), 0
                    );
                }
            }
            false => {
                #[cfg(target_arch = "aarch64")]
                #[allow(clippy::useless_conversion)]
                {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; ldr W(**host_reg), [x0, offset]
                    );
                }
            }
        }
    }
    pub fn emit_load_reg_stackless(&mut self, guest_reg: u8) -> LoadedReg<AllocResultStackless> {
        let host_reg = self.alloc_reg_stackless(guest_reg);
        match host_reg.result {
            Err(RegAllocErrorStackless::AlreadyAllocatedTo(_)) => {}
            _ => self.impl_emit_reg_load(guest_reg, &host_reg),
        }
        host_reg
    }

    #[allow(clippy::useless_conversion)]
    fn emit_immediate_large(&mut self, guest_reg: Guest, imm: u32) -> EmitSummary {
        let reg = self.alloc_reg(guest_reg);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            self.asm
            ; .arch aarch64
            ; movz W(*reg), imm >> 16, LSL #16
            ; movk W(*reg), imm & 0x0000_ffff
        );

        self.mark_dirty(guest_reg);
        reg.restore(self);

        EmitSummary::default()
    }

    #[allow(clippy::useless_conversion)]
    fn emit_immediate_sext(&mut self, guest_reg: Guest, imm: i16) -> EmitSummary {
        let reg = self.alloc_reg(guest_reg);
        self.emit_imm16_sext(reg.reg(), imm);
        self.mark_dirty(guest_reg);
        reg.restore(self);
        EmitSummary::default()
    }

    #[allow(clippy::useless_conversion)]
    fn emit_immediate_uext(&mut self, guest_reg: Guest, imm: i16) -> EmitSummary {
        let reg = self.alloc_reg(guest_reg);
        self.emit_imm16_uext(reg.reg(), imm);
        self.mark_dirty(guest_reg);
        reg.restore(self);
        EmitSummary::default()
    }

    fn emit_zero(&mut self, guest_reg: Guest) -> EmitSummary {
        self.emit_immediate_uext(guest_reg, 0)
    }

    #[allow(clippy::useless_conversion)]
    fn emit_load_and_move_into(&mut self, target: Guest, reg: Guest) -> EmitSummary {
        let rd = self.alloc_reg(target);
        self.emit_load_temp_reg(reg, Reg::W(1));
        dynasm!(
            self.asm
            ; .arch aarch64
            ; mov W(*rd), w1
        );
        self.mark_dirty(target);
        rd.restore(self);
        EmitSummary::default()
    }

    fn mark_dirty(&mut self, guest_reg: u8) {
        self.reg_alloc.dirty.set(guest_reg as usize, true);
    }

    fn mark_clean(&mut self, guest_reg: u8) {
        self.reg_alloc.dirty.set(guest_reg as usize, false);
    }

    #[allow(clippy::useless_conversion)]
    fn emit_write_pc(&mut self, temp_reg: Reg, new_pc: u32) {
        let n = temp_reg.to_idx();
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.asm
            ; .arch aarch64
            ; movz W(n), new_pc >> 16 , LSL #16
            ; movk W(n), new_pc & 0x0000_FFFF
            ; str W(n),  [x0, Emu::<A>::PC_OFFSET as _]
        )
    }

    #[allow(clippy::useless_conversion)]
    fn emit_save_volatile_registers(&mut self) -> SmallVec<[u8; 32]> {
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            self.asm
            ; .arch aarch64
            ; str x0, [sp, #-16]!
        );
        self.reg_alloc
            .allocated_volatile()
            .into_iter()
            .inspect(|reg| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; .arch aarch64
                    ; str X(*reg), [sp, #-16]!
                );
            })
            .collect()
    }

    #[allow(clippy::useless_conversion)]
    fn emit_restore_saved_registers(&mut self, saved: impl DoubleEndedIterator<Item = u8>) {
        saved.into_iter().rev().for_each(|reg| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                self.asm
                ; .arch aarch64
                ; ldr X(reg), [sp], #16
            );
        });
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            self.asm
            ; .arch aarch64
            ; ldr x0, [sp], #16
        );
    }
}

#[derive(Debug, Clone, d::Deref, d::DerefMut)]
pub struct LoadedReg<T> {
    result:   T,
    reg:      Reg,
    #[deref]
    #[deref_mut]
    reg_idx:  u8,
    restored: Cell<bool>,
}

impl From<AllocResult> for LoadedReg<AllocResult> {
    fn from(result: AllocResult) -> Self {
        let reg = match &result {
            Ok(reg) => **reg,
            Err(
                RegAllocError::EvictToMemory(_, reg)
                | RegAllocError::EvictToStack(_, reg)
                | RegAllocError::AlreadyAllocatedTo(reg),
            ) => **reg,
        };
        Self {
            reg,
            reg_idx: reg.to_idx(),
            result,
            restored: Cell::new(false),
        }
    }
}
impl From<AllocResultStackless> for LoadedReg<AllocResultStackless> {
    fn from(result: AllocResultStackless) -> Self {
        let reg = match &result {
            Ok(reg) => **reg,
            Err(
                RegAllocErrorStackless::EvictToMemory(_, reg)
                | RegAllocErrorStackless::AlreadyAllocatedTo(reg),
            ) => **reg,
        };
        Self {
            reg,
            reg_idx: reg.to_idx(),
            result,
            restored: Cell::new(false),
        }
    }
}

impl LoadedReg<AllocResult> {
    fn reg(&self) -> Reg {
        self.reg
    }
    fn restore<A: Allocator + Copy>(&self, dynarec: &mut Dynarec<A>) {
        debug_assert!(!self.restored.get(), "loaded reg already restored.");

        self.restored.set(true);

        if let Err(RegAllocError::EvictToStack(_, reg)) = self.result {
            let current_guest = dynarec.reg_alloc.reverse_mapping[(*reg).to_idx() as usize];
            if dynarec.reg_alloc.dirty[current_guest as usize] {
                dynarec.emit_writeback(current_guest, *reg);
            }

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                dynarec.asm
                ; .arch aarch64
                ; ldr W(reg), [sp], #16
            )
        }
    }
}

#[derive(d::Debug)]
pub struct ScheduledEmitter<A: Allocator + Copy> {
    #[debug(skip)]
    pub(crate) emitter:  DynEmitter<A>,
    #[debug("{}", hex(self.schedule))]
    pub(crate) schedule: u32,
    pub(crate) pc:       u32,
}

impl<A: Allocator + Copy> PartialEq for ScheduledEmitter<A> {
    fn eq(&self, other: &Self) -> bool {
        self.schedule == other.schedule
    }
}
impl<A: Allocator + Copy> Eq for ScheduledEmitter<A> {}

impl<A: Allocator + Copy> PartialOrd for ScheduledEmitter<A> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<A: Allocator + Copy> Ord for ScheduledEmitter<A> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.schedule.cmp(&other.schedule)
    }
}

#[derive(d::Debug)]
pub struct Scheduler<A: Allocator + Copy> {
    pub(crate) queue: heapless::BinaryHeap<ScheduledEmitter<A>, Min, 4>,
}

unsafe impl<A: Allocator + Copy> Send for Scheduler<A> {}
unsafe impl<A: Allocator + Copy> Sync for Scheduler<A> {}

impl<A: Allocator + Copy> Default for Scheduler<A> {
    fn default() -> Self {
        Self {
            queue: Default::default(),
        }
    }
}

impl<A: Allocator + Copy> Dynarec<A> {
    pub fn lock_register(&mut self, reg: &LoadedReg<AllocResultStackless>) {
        debug_assert!(self.reg_alloc.allocatable[**reg as usize]);
        self.reg_alloc.evict_at(reg.reg);
        self.reg_alloc.allocatable.set(**reg as _, false);
    }
    pub fn unlock_register(&mut self, reg: &LoadedReg<AllocResultStackless>) {
        debug_assert!(!self.reg_alloc.allocatable[**reg as usize]);
        self.reg_alloc.allocatable.set(**reg as _, true);
    }
    pub fn pop_scheduled_at<'a, 'd>(&mut self, pc: u32) -> Option<ScheduledEmitter<A>> {
        if let Some(emitter) = self.scheduler.queue.peek() {
            if emitter.schedule <= pc {
                return self.scheduler.queue.pop();
            }
        }
        None
    }
}

// impl Drop for LoadedReg {
//     fn drop(&mut self) {
//         if self.armed {
//             debug_assert!(
//                 self.restored.get(),
//                 "loaded register handle dropped without calling restore."
//             )
//         }
//     }
// }

impl<A: Allocator + Copy> Emu<A> {
    fn linear_fetch(&self) -> impl Iterator<Item = (OpCode, DecodedOp)> {
        let mut iter = self
            .linear_fetch_no_decode()
            .map(|op| (op, DecodedOp::new(op)));
        let mut taking: Option<i32> = None;
        iter::from_fn(move || {
            taking = taking.map(|x| x - 1);
            if matches!(taking, Some(0)) {
                return None;
            }

            let value = iter.next();
            if let Some((_, op)) = value {
                if op.is_boundary() {
                    taking = Some(2);
                }
            }
            value
        })
    }
    fn linear_fetch_no_decode(&self) -> impl Iterator<Item = OpCode> {
        (self.cpu.pc..)
            // .step_by(0x4)
            .step_by(max_simd_elements::<u32>() * size_of::<u32>())
            .flat_map(|address| self.read_pure::<Simd<u32, 4>>(address).to_array())
            // .map(|address| self.read(address))
            .map(OpCode::new_with_raw_value)
    }
}

pub fn run_step<A: Allocator + Copy + Clone>(emu: &mut Emu<A>, dynarec: &mut Dynarec<A>) {
    let pc = emu.cpu.pc;
    let block = match emu.dynarec_cache.remove(pc) {
        None => {
            let block = fetch_and_compile_single_threaded(emu, dynarec).unwrap();
            emu.stats.blocks_compiled += 1;
            #[cfg(debug_assertions)]
            {
                INSTR_COMPILED.fetch_add(block.op_count as u64, Ordering::Relaxed);
                BLOCKS_COMPILED.fetch_add(1, Ordering::Relaxed);
                CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            }
            block
        }
        Some(block) => {
            #[cfg(debug_assertions)]
            CACHE_HITS.fetch_add(1, Ordering::Relaxed);

            block
        }
    };
    block(emu, false);
    emu.stats.blocks_ran += 1;

    dynarec.last_ran_function = Some(block.function.clone());
    emu.dynarec_cache.insert(pc, block);
}

#[derive(strum::EnumCount, strum::EnumDiscriminants, strum::EnumIs)]
#[strum_discriminants(derive(
    strum::Display,
    strum::VariantArray,
    strum::EnumCount,
    strum::EnumIter
))]
#[strum_discriminants(name(PipelineV2Stage))]
#[strum_discriminants(repr(u8))]
pub enum PipelineV2<A: Allocator + Copy> {
    Uninit,
    Init {
        dynarec: Box<Dynarec<A>>,
        pc:      u32,
    },
    Compiled {
        pc:        u32,
        func:      DynarecBlock<A>,
        dynarec:   Option<Box<Dynarec<A>>>,
        scheduler: Option<Box<Scheduler<A>>>,
    },
    Called {
        pc:        u32,
        times:     usize,
        func:      DynarecBlock<A>,
        dynarec:   Option<Box<Dynarec<A>>>,
        scheduler: Option<Box<Scheduler<A>>>,
    },
    Cached {
        dynarec:   Option<Box<Dynarec<A>>>,
        scheduler: Option<Box<Scheduler<A>>>,
    },
}

#[derive(thiserror::Error, Debug)]
#[error("failed to compile block.")]
pub struct PipelineCompileError;

pub(crate) fn fetch_and_compile_single_threaded<A: Allocator + Copy>(
    emu: &Emu<A>,
    dynarec: &mut Dynarec<A>,
) -> Result<DynarecBlock<A>, PipelineCompileError> {
    dynarec.emit_block_prelude();
    let initial_pc = emu.cpu.pc;
    let mut lifetime = 2u8;
    let mut cycles = 0u32;
    let mut scratch_cursor = 0;
    let mut op_count = 0;
    let mut last_address = initial_pc;
    let mut pc_updated = false;
    emu.linear_fetch_no_decode()
        .zip((initial_pc..).step_by(0x4))
        .map_while(|(op, address)| {
            let [op] = DecodedOp::decode([op]);
            let ret = match lifetime {
                2 => Some((op, address)),
                1 => {
                    lifetime -= 1;
                    Some((op, address))
                }
                0 => None,
                _ => unreachable!(),
            };
            let cache_boundary_check = address >> 2;
            if cache_boundary_check != 0 && cache_boundary_check.is_multiple_of(PAGE_LEN as u32) {
                lifetime = 1
            }
            if op.is_boundary() {
                lifetime = 1;
                if op.is_hard_boundary() {
                    lifetime = 0;
                }
            }
            tracing::trace!(?ret);
            ret
        })
        .for_each(|(op, address)| {
            pc_updated |= op
                .emit(EmitCtx {
                    dynarec,
                    pc: address,
                    d_clock: cycles,
                    delay_slot: false,
                    scratch_cursor: &mut scratch_cursor,
                })
                .pc_updated;
            if let Some(pre_scheduled) = dynarec.pop_scheduled_at(address) {
                // let cache_boundary_check = (pre_scheduled.pc.saturating_sub(0x4)) >> 2;
                // let boundary = cache_boundary_check != 0
                //     && cache_boundary_check.is_multiple_of(PAGE_LEN as u32);
                // assert!(!boundary, "delay slot is on cache boundary");
                pc_updated |= pre_scheduled
                    .emitter
                    .call((EmitCtx {
                        dynarec,
                        pc: pre_scheduled.pc,
                        d_clock: cycles,
                        delay_slot: true,
                        scratch_cursor: &mut scratch_cursor,
                    },))
                    .pc_updated;
            }
            cycles += op.cycles() as u32;
            if op.hazard() != 0 {
                cycles -= 1;
            }
            last_address = address;
            op_count += 1;
        });

    // drain scheduler
    while let Some(emitter) = dynarec.scheduler.queue.pop() {
        tracing::trace!("draining {:?}", emitter);
        pc_updated |= emitter
            .emitter
            .call((EmitCtx {
                dynarec,
                pc: emitter.pc,
                d_clock: cycles,
                // this happens in the delay slot basically
                delay_slot: true,
                scratch_cursor: &mut scratch_cursor,
            },))
            .pc_updated;
    }

    let new_pc = match pc_updated {
        true => None,
        false => Some(last_address + 0x4),
    };
    dynarec.emit_block_epilogue(cycles, new_pc, true);

    let func = dynarec.finalize().map_err(|_| PipelineCompileError)?;

    Ok(DynarecBlock {
        function: func,
        op_count,
        pc: initial_pc,
    })
}

const CACHE_LEN: usize = (kb(2048) + kb(512)) >> 2;
const PAGE_COUNT: usize = CACHE_LEN / PAGE_LEN;
const PAGE_LEN: usize = kb(16);

/// # DynarecCache
///
/// maps the entire psx ram and bios losslessly into a a flat, paged, ~320kb buffer
#[derive(derive_more::Debug, Clone)]
pub struct DynarecCache<A: Allocator + Copy> {
    blocks:   Box<[Option<DynarecBlock<A>>; PAGE_LEN * PAGE_COUNT], A>,
    metadata: [CachePage; PAGE_COUNT],
}

#[derive(derive_more::Debug, Clone)]
struct CachePage {
    inserted: HashSet<usize>,
    cleared:  bool,
}

impl Default for CachePage {
    fn default() -> Self {
        Self {
            cleared:  true,
            inserted: HashSet::new(),
        }
    }
}

impl<A: Allocator + Copy> DynarecCache<A> {
    pub fn new(alloc: A) -> Self {
        let mut buf = Box::new_uninit_slice_in(PAGE_LEN * PAGE_COUNT, alloc);
        for el in buf.iter_mut() {
            *el = MaybeUninit::new(None);
        }
        let buf = unsafe { buf.assume_init() }.into_array().ok();
        Self {
            metadata: core::array::from_fn(|_| CachePage::default()),
            blocks:   buf.unwrap(),
        }
    }
}

impl<A: Allocator + Copy> DynarecCache<A> {
    const PROB: Option<usize> = Self::map_addr_to_idx(0x8004f434);
    const RAM_END: usize = Self::map_addr_to_idx(0x200000).unwrap();
    const BIOS_START: usize = Self::map_addr_to_idx(0xbfc0_0000).unwrap();

    fn block_mut(&mut self, page_idx: usize, element_idx: usize) -> &mut Option<DynarecBlock<A>> {
        &mut self.blocks[page_idx * PAGE_LEN + element_idx]
    }
    fn block(&self, page_idx: usize, element_idx: usize) -> &Option<DynarecBlock<A>> {
        &self.blocks[page_idx * PAGE_LEN + element_idx]
    }

    const fn map_addr_to_idx(address: u32) -> Option<usize> {
        match address & 0x1fff_ffff {
            // align by 4
            // ram
            addr @ 0..0x200000 => Some(((addr as usize) >> 2) / PAGE_LEN),
            // bios
            addr @ 0x1fc00000.. => Some(((addr as usize - 0x1fc00000 + kb(2048)) >> 2) / PAGE_LEN),
            _ => None,
        }
    }
    const fn map_addr(address: u32) -> Option<(usize, usize)> {
        let page_idx = Self::map_addr_to_idx(address)?;
        let element_idx = (address & 0xffff) >> 2;
        Some((page_idx, element_idx as usize))
    }
    pub fn remove(&mut self, at: u32) -> Option<DynarecBlock<A>> {
        Self::map_addr(at)
            .map(|(page_idx, element_idx)| self.block_mut(page_idx, element_idx))
            .and_then(|block| block.take())
    }
    pub fn get(&self, at: u32) -> Option<&DynarecBlock<A>> {
        Self::map_addr(at)
            .and_then(|(page_idx, element_idx)| self.block(page_idx, element_idx).as_ref())
    }
    pub fn insert(&mut self, at: u32, value: DynarecBlock<A>) -> bool {
        if let Some((page_idx, element_idx)) = Self::map_addr(at) {
            *self.block_mut(page_idx, element_idx) = Some(value);
            self.metadata[page_idx].cleared = false;
            self.metadata[page_idx].inserted.insert(element_idx);
            true
        } else {
            false
        }
    }
    pub fn invalidate(&mut self, at: u32) {
        if let Some((page_idx, _)) = Self::map_addr(at) {
            if self.metadata[page_idx].cleared {
                return;
            }
            let page_blocks = &self.metadata[page_idx].inserted;
            for block in page_blocks {
                self.blocks[page_idx * PAGE_LEN + *block] = None;
            }
            self.metadata[page_idx].cleared = true;
            self.metadata[page_idx].inserted.clear();
        }
    }
    pub extern "C" fn jump(emu: &mut Emu<A>, at: u32) -> Option<NonNull<*const fn(*mut Emu<A>)>> {
        tracing::info!("jump called: pc={}", hex(at));
        unsafe {
            emu.dynarec_cache.get(at).map(|block| {
                tracing::info!("  found block at {:p}", block.function.func);
                NonNull::new_unchecked(block.function.func as _)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Global;

    use color_eyre::Result;
    use pchan_utils::setup_tracing;

    use super::*;
    use crate::Emu;

    #[test]
    fn dynarec_minimal_test() -> Result<()> {
        setup_tracing();
        let mut emu = Emu::default();
        let mut dynarec = Dynarec::new(CreateDynarecParams {
            reg_alloc: None,
            scheduler: None,
            asm:       None,
            alloc:     Global,
        });

        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(dynarec.asm; .arch aarch64; ret);
        }

        let func = dynarec.finalize()?;
        tracing::info!("Calling JIT function...");
        func.func.call((&mut emu,));
        tracing::info!("JIT call succeeded!");

        Ok(())
    }
}
