use color_eyre::eyre::bail;
use dynasm::dynasm;
use dynasmrt::Assembler;
use dynasmrt::DynasmApi;
use dynasmrt::DynasmLabelApi;
use dynasmrt::ExecutableBuffer;
use flume::Receiver;
use flume::Sender;
use pchan_utils::hex;
use smallbox::SmallBox;
use smallvec::SmallVec;
use std::cell::Cell;
use std::collections::VecDeque;
use std::hash::Hasher;
use std::ops::Deref;
use std::simd::Simd;
use std::sync::Arc;
use std::sync::LazyLock;
use tracing::Instrument;
use tracing::Level;
use tracing::enabled;

use crate::Emu;
use crate::cpu::exceptions::Exceptions;
use crate::cpu::ops::OpCode;
use crate::cpu::reg_str;
use crate::dynarec_v2::emitters::DecodedOpNew;
use crate::dynarec_v2::emitters::DynarecOp;
use crate::dynarec_v2::emitters::EmitCtx;
use crate::dynarec_v2::emitters::EmitSummary;
use crate::dynarec_v2::regalloc::*;
use crate::io::IO;
use crate::max_simd_elements;

pub mod emitters;
pub mod regalloc;

#[cfg(feature = "fetch-channel")]
pub type FetchedOp = (u32, DecodedOpNew);
#[cfg(feature = "fetch-channel")]
pub static FETCH_CHANNEL: LazyLock<(Sender<FetchedOp>, Receiver<FetchedOp>)> =
    LazyLock::new(|| flume::bounded(1024));

#[cfg(target_arch = "aarch64")]
type Reloc = dynasmrt::aarch64::Aarch64Relocation;
#[cfg(target_arch = "x86_64")]
type Reloc = dynasmrt::x64::X64Relocation;

type DynEmitter = SmallBox<dyn Fn(EmitCtx) -> EmitSummary, [u8; 64]>;

#[derive(derive_more::Debug)]
pub struct Dynarec {
    reg_alloc: RegAlloc,
    #[debug("{}", self.delay_queue.len())]
    delay_queue: VecDeque<DynEmitter>,
    #[debug(skip)]
    asm: Assembler<Reloc>,
}

unsafe impl Send for Dynarec {}
unsafe impl Sync for Dynarec {}

impl Default for Dynarec {
    fn default() -> Self {
        let asm = Assembler::new_with_capacity(size_of::<u64>() * 128)
            .expect("failed to create assembler");
        Self {
            delay_queue: VecDeque::with_capacity(2),
            reg_alloc: Default::default(),
            asm,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynarecFunction {
    func: fn(*mut Emu),
    exec: Arc<ExecutableBuffer>,
}

#[derive(Debug, Clone)]
pub struct DynarecBlock {
    function: DynarecFunction,
    hash: u64,
    op_count: usize,
    is_loop: bool,
}

type DynarecBlockArgs<'a> = (&'a mut Emu, bool);

impl DynarecBlock {
    pub fn call_block(&self, (emu, instrument): DynarecBlockArgs) {
        // reset delta clock before running
        emu.cpu.d_clock = 0;

        if instrument {
            (self.function.func)
                .instrument(tracing::info_span!("fn", addr = ?self.function.func))
                .inner()(emu)
        } else {
            (self.function.func)(emu)
        };

        emu.run_io();
    }
}

impl FnMut<DynarecBlockArgs<'_>> for DynarecBlock {
    extern "rust-call" fn call_mut(&mut self, args: DynarecBlockArgs) -> Self::Output {
        self.call_block(args)
    }
}

impl FnOnce<DynarecBlockArgs<'_>> for DynarecBlock {
    type Output = ();
    extern "rust-call" fn call_once(mut self, args: DynarecBlockArgs) -> Self::Output {
        self.call_mut(args)
    }
}

impl Fn<DynarecBlockArgs<'_>> for DynarecBlock {
    extern "rust-call" fn call(&self, args: DynarecBlockArgs) -> Self::Output {
        self.call_block(args);
    }
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

        if enabled!(Level::DEBUG) {
            use std::fs::File;
            use std::io::Write;
            File::create("/tmp/jit_code.bin")?.write_all(exec.as_ref())?;
            tracing::debug!("Wrote {} bytes to /tmp/jit_code.bin", exec.len());
        }

        let func = unsafe { std::mem::transmute::<*const u8, fn(_)>(exec.as_ptr()) };
        Ok(DynarecFunction {
            func,
            exec: Arc::new(exec),
        })
    }
    fn emit_block_prelude(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(
                self.asm
                ; .arch aarch64
                ; b >after_table

                // -- function table --
                ; -> write32v2:
                ; .u64 Emu::write32v2 as *const () as _
                ; -> write16v2:
                ; .u64 Emu::write16v2 as *const () as _
                ; -> write8v2:
                ; .u64 Emu::write8v2 as *const () as _
                ; -> readi8v2:
                ; .u64 Emu::readi8v2 as *const () as _
                ; -> readu8v2:
                ; .u64 Emu::readu8v2 as *const () as _
                ; -> readi16v2:
                ; .u64 Emu::readi16v2 as *const () as _
                ; -> readu16v2:
                ; .u64 Emu::readu16v2 as *const () as _
                ; -> read32v2:
                ; .u64 Emu::read32v2 as *const () as _
                ; -> handle_syscall:
                ; .u64 Emu::handle_syscall as *const () as _
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
        let offset = Emu::reg_offset(guest_reg) as u32;

        if enabled!(Level::TRACE) {
            tracing::trace!("store: guest r{}", guest_reg);
        }

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

    fn emit_writeback_pair_free(asm: &mut Assembler<Reloc>, arr: [(u8, Reg); 2]) {
        let [(gr1, hr1), (gr2, hr2)] = arr;
        debug_assert!(
            hr1.consecutive(hr2),
            "pairs store must be of consecutive registers"
        );

        let offset = Emu::reg_offset(gr1) as u32;

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

    fn emit_block_epilogue(&mut self, d_clock: u32, new_pc: Option<u32>) {
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

        const {
            debug_assert!(Emu::D_CLOCK_OFFSET == Emu::PC_OFFSET + 4);
        };

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
                    ; stp w24, w25, [x0, Emu::PC_OFFSET as _]
                )
            }
            None => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; .arch aarch64
                    ; movz w25, d_clock >> 16 , LSL #16
                    ; movk w25, d_clock & 0x0000_FFFF
                    ; str w25, [x0, Emu::D_CLOCK_OFFSET as _]
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
                ; ret
            )
        }
    }
    fn alloc_reg(&mut self, guest_reg: u8) -> LoadedReg {
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
                    ; str W(host_reg), [sp], #16
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
            let offset = Emu::reg_offset(guest_reg) as u32;
            dynasm!(
                self.asm
                ; .arch aarch64
                ; ldr W(host_reg), [x0, offset]
            );
        }
    }
    fn emit_load_reg(&mut self, guest_reg: u8) -> LoadedReg {
        let offset = Emu::reg_offset(guest_reg) as u32;

        let host_reg = self.alloc_reg(guest_reg);
        match host_reg.result {
            Err(RegAllocError::AlreadyAllocatedTo(_)) => {}
            _ => {
                if guest_reg == 0 {
                    #[cfg(target_arch = "aarch64")]
                    #[allow(clippy::useless_conversion)]
                    {
                        dynasm!(
                            self.asm
                            ; .arch aarch64
                            ; mov W(*host_reg), 0
                        );
                    }
                } else {
                    #[cfg(target_arch = "aarch64")]
                    #[allow(clippy::useless_conversion)]
                    {
                        dynasm!(
                            self.asm
                            ; .arch aarch64
                            ; ldr W(*host_reg), [x0, offset]
                        );
                    }
                }
            }
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

    fn set_delay_slot(&mut self, emitter: impl Fn(EmitCtx) -> EmitSummary + 'static) {
        self.delay_queue.push_back(SmallBox::new(emitter) as _);
    }

    #[allow(clippy::useless_conversion)]
    fn emit_write_pc(&mut self, temp_reg: Reg, new_pc: u32) {
        let n = temp_reg.to_idx();
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.asm
            ; .arch aarch64
            ; movz W(n), new_pc >> 16 , LSL #16
            ; movk W(n), new_pc & 0x0000_FFFF
            ; str W(n),  [x0, Emu::PC_OFFSET as _]
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
        // for reg in self.reg_alloc.allocated_volatile().chunks(2) {
        //     match reg {
        //         [a, b] => {
        //             #[cfg(target_arch = "aarch64")]
        //             dynasm!(
        //                 self.asm
        //                 ; .arch aarch64
        //                 ; stp X(*a), X(*b), [sp, #-16]!
        //             );
        //         }
        //         [reg] => {
        //             #[cfg(target_arch = "aarch64")]
        //             dynasm!(
        //                 self.asm
        //                 ; .arch aarch64
        //                 ; str X(*reg), [sp, #-16]!
        //             );
        //         }
        //         _ => unreachable!(),
        //     };
        // }
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
        // for reg in self.reg_alloc.allocated_volatile().chunks(2).rev() {
        //     match reg {
        //         [a, b] => {
        //             #[cfg(target_arch = "aarch64")]
        //             dynasm!(
        //                 self.asm
        //                 ; .arch aarch64
        //                 ; ldp X(*a), X(*b), [sp], #16
        //             );
        //         }
        //         [reg] => {
        //             #[cfg(target_arch = "aarch64")]
        //             dynasm!(
        //                 self.asm
        //                 ; .arch aarch64
        //                 ; ldr X(*reg), [sp], #16
        //             );
        //         }
        //         _ => unreachable!(),
        //     };
        // }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedReg {
    result: AllocResult,
    reg: Reg,
    reg_idx: u8,
    restored: Cell<bool>,
}

impl From<AllocResult> for LoadedReg {
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

impl Deref for LoadedReg {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl AsRef<u8> for LoadedReg {
    fn as_ref(&self) -> &u8 {
        &self.reg_idx
    }
}

impl AsRef<Reg> for LoadedReg {
    fn as_ref(&self) -> &Reg {
        &self.reg
    }
}

impl LoadedReg {
    fn reg(&self) -> Reg {
        *self.as_ref()
    }
    fn restore(&self, dynarec: &mut Dynarec) {
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

impl Emu {
    fn linear_fetch(&self) -> impl Iterator<Item = (OpCode, DecodedOpNew)> {
        let mut iter = self
            .linear_fetch_no_decode()
            .map(|op| (op, DecodedOpNew::new(op)));
        let mut taking: Option<i32> = None;
        std::iter::from_fn(move || {
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
            .flat_map(|address| self.read::<Simd<u32, 4>>(address).to_array())
            // .map(|address| self.read(address))
            .map(OpCode)
    }
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
pub enum PipelineV2 {
    Uninit,
    Init {
        dynarec: Box<Dynarec>,
        pc: u32,
    },
    Compiled {
        pc: u32,
        func: DynarecBlock,
        dynarec: Option<Box<Dynarec>>,
    },
    Called {
        pc: u32,
        times: usize,
        func: DynarecBlock,
        dynarec: Option<Box<Dynarec>>,
    },
    Cached {
        dynarec: Option<Box<Dynarec>>,
    },
}

impl PipelineV2 {
    pub fn new(emu: &Emu) -> Self {
        Self::Init {
            dynarec: Box::new(Dynarec::default()),
            pc: emu.cpu.pc,
        }
    }
    pub fn stage(&self) -> PipelineV2Stage {
        self.into()
    }
    pub fn run_once(mut self, emu: &mut Emu) -> color_eyre::Result<Self> {
        for _ in 0..3 {
            self = self.step(emu)?;
        }
        Ok(self)
    }
    pub fn step(self, emu: &mut Emu) -> color_eyre::Result<Self> {
        match self {
            PipelineV2::Init { pc, dynarec } => match emu.dynarec_cache.remove(&pc) {
                None => {
                    let func = fetch_and_compile_single_threaded(emu, dynarec)?;
                    Ok(PipelineV2::Compiled {
                        pc,
                        func,
                        dynarec: None,
                    })
                }
                Some(block) => {
                    let hash = fast_hash(emu, block.op_count);
                    if hash == block.hash {
                        Ok(PipelineV2::Compiled {
                            pc,
                            func: block,
                            dynarec: Some(dynarec),
                        })
                    } else {
                        let func = fetch_and_compile_single_threaded(emu, dynarec)?;
                        Ok(PipelineV2::Compiled {
                            pc,
                            func,
                            dynarec: None,
                        })
                    }
                }
            },
            PipelineV2::Compiled {
                pc,
                mut func,
                dynarec,
            } => {
                func(emu, false);
                let mut count = 1;
                while emu.cpu.pc == pc {
                    func.is_loop = true;
                    func(emu, false);
                    count += 1;
                }
                Ok(PipelineV2::Called {
                    times: count,
                    pc,
                    func,
                    dynarec,
                })
            }
            PipelineV2::Called {
                pc, func, dynarec, ..
            } => {
                emu.dynarec_cache.insert(pc, func);
                Ok(PipelineV2::Cached { dynarec })
            }
            PipelineV2::Cached { dynarec } => Ok(PipelineV2::Init {
                pc: emu.cpu.pc,
                dynarec: dynarec.unwrap_or_else(|| Box::new(Dynarec::default())),
            }),
            PipelineV2::Uninit => Ok(PipelineV2::Uninit),
        }
    }
}

fn fast_hash(emu: &Emu, op_count: usize) -> u64 {
    let mut hasher = rapidhash::fast::RapidHasher::default();
    emu.linear_fetch_no_decode()
        .take(op_count)
        .map(|op| op.0)
        .for_each(|op| hasher.write_u32(op));

    hasher.finish()
}

fn fetch_and_compile_single_threaded(
    emu: &Emu,
    mut dynarec: Box<Dynarec>,
) -> color_eyre::Result<DynarecBlock> {
    dynarec.emit_block_prelude();

    let mut hasher = rapidhash::fast::RapidHasher::default();

    type FetchItem = (OpCode, DecodedOpNew);
    #[derive(Debug)]
    struct FetchState {
        op_count: usize,
        cycles: u32,
        pc: u32,
        window: [Option<FetchItem>; 2],
        hit_boundary: bool,
        pc_updated: bool,
    }

    impl FetchState {
        const fn push_item(&mut self, item: Option<FetchItem>) {
            if self.window[0].is_some() {
                unsafe {
                    self.window.swap_unchecked(0, 1);
                }
            }
            self.window[0] = item;
        }
        const fn pop_item(&mut self) -> Option<FetchItem> {
            let item = self.window[1].take();
            unsafe {
                self.window.swap_unchecked(0, 1);
            }
            item
        }
        const fn back(&self) -> Option<&FetchItem> {
            self.window[1].as_ref()
        }
        const fn clear_items(&mut self) {
            self.window[0] = None;
            self.window[1] = None;
        }
        const fn apply(&mut self, summary: EmitSummary) {
            self.pc_updated |= summary.pc_updated;
        }
    }

    let initial_pc = emu.cpu.pc;
    let mut state = FetchState {
        op_count: 0,
        cycles: 0,
        pc: initial_pc,
        window: [None; 2],
        hit_boundary: false,
        pc_updated: false,
    };

    let mut iter = emu
        .linear_fetch_no_decode()
        .map(|op| (op, DecodedOpNew::new(op)));

    state.push_item(iter.next());
    state.push_item(iter.next());
    loop {
        let Some((opcode, op)) = state.pop_item() else {
            break;
        };

        state.pc = initial_pc + state.op_count as u32 * 0x4;

        state.cycles += op.cycles() as u32;
        state.op_count += 1;

        if let Some((_, next)) = state.back() {
            state.cycles -= next.cycles().min(op.hazard()) as u32;
        }
        hasher.write_u32(opcode.0);

        if op.is_hard_boundary() {
            state.clear_items();
            state.hit_boundary = true;
        } else if op.is_boundary() {
            state.hit_boundary = true;
        } else if !state.hit_boundary {
            state.push_item(iter.next());
        }

        let delay_slot = dynarec.delay_queue.pop_front();

        state.apply(op.emit(EmitCtx {
            dynarec: &mut dynarec,
            pc: state.pc,
        }));

        #[cfg(feature = "fetch-channel")]
        {
            let _ = FETCH_CHANNEL.0.send((state.pc, op));
        }

        if let Some(emitter) = delay_slot {
            let summary = emitter(EmitCtx {
                dynarec: &mut dynarec,
                // -4 because delayed effects think they are the same instruction
                pc: state.pc - 0x4,
            });
            state.apply(summary);
        }

        tracing::trace!(pc = hex(state.pc), %op);
    }

    state.pc = initial_pc + (state.op_count as u32) * 0x4;

    if let Some(emitter) = dynarec.delay_queue.pop_front() {
        state.apply(emitter(EmitCtx {
            dynarec: &mut dynarec,
            pc: state.pc,
        }));
    }

    if enabled!(Level::TRACE) {
        tracing::trace!(?state.op_count);
    }

    dynarec.emit_block_epilogue(state.cycles, (!state.pc_updated).then_some(state.pc));

    let func = dynarec.finalize()?;
    let hash = hasher.finish();

    Ok(DynarecBlock {
        function: func,
        op_count: state.op_count,
        is_loop: false,
        hash,
    })
}

#[cfg(test)]
mod tests {

    extern crate test;

    use color_eyre::Result;
    use pchan_utils::setup_tracing;

    use super::*;
    use crate::Emu;

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
        func.func.call((&mut emu,));
        tracing::info!("JIT call succeeded!");

        Ok(())
    }
}
