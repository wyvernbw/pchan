use bitvec::prelude as bv;
use color_eyre::eyre::bail;
use dynasm::dynasm;
use dynasmrt::Assembler;
use dynasmrt::DynasmApi;
use dynasmrt::ExecutableBuffer;
use pchan_utils::array;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::offset_of;
use std::rc::Rc;
use std::simd::Simd;
use tracing::Instrument;
use tracing::Level;
use tracing::enabled;

use crate::Emu;
use crate::cpu::ops::OpCode;
use crate::cpu::reg_str;
use crate::dynarec_v2::dynarec_ops::DecodedOpNew;
use crate::dynarec_v2::dynarec_ops::DynarecOp;
use crate::dynarec_v2::dynarec_ops::EmitCtx;
use crate::dynarec_v2::dynarec_ops::EmitSummary;
use crate::dynarec_v2::dynarec_ops::ResultIntoInner;
use crate::io::IO;
use crate::max_simd_elements;
use crate::memory::ext;
use crate::{cpu::Cpu, memory::Memory};

pub mod dynarec_ops;

#[cfg(target_arch = "aarch64")]
type Reloc = dynasmrt::aarch64::Aarch64Relocation;
#[cfg(target_arch = "x86_64")]
type Reloc = dynasmrt::x64::X64Relocation;

#[derive(Debug)]
pub struct Dynarec {
    reg_alloc: RegAlloc,
    delay_slot: Option<fn(EmitCtx) -> EmitSummary>,
    asm: Assembler<Reloc>,
}

impl Default for Dynarec {
    fn default() -> Self {
        let asm = Assembler::new_with_capacity(size_of::<u32>() * 256)
            .expect("failed to create assembler");
        Self {
            delay_slot: None,
            reg_alloc: Default::default(),
            asm,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct RegAlloc {
    loaded: bv::BitArray<[u32; 1]>,
    dirty: bv::BitArray<[u32; 1]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reg {
    #[cfg(target_arch = "aarch64")]
    W(u8),
}

#[cfg(target_arch = "aarch64")]
impl Reg {
    const WZR: Self = Reg::W(31);
    const WSP: Self = Reg::W(31);

    const DELAY_1: Self = Reg::W(28);

    pub fn consecutive(self: Reg, b: Reg) -> bool {
        let Reg::W(w) = self;
        b == Reg::W(w + 1)
    }
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

#[derive(Debug, Clone)]
pub struct DynarecFunction {
    func: fn(*mut Cpu, *mut Memory),
    exec: Rc<ExecutableBuffer>,
}

#[derive(Debug, Clone)]
pub struct DynarecBlock {
    function: DynarecFunction,
    hash: u64,
    op_count: usize,
}

type DynarecBlockArgs<'a> = (&'a mut Emu, bool);

impl DynarecBlock {
    pub fn call_block(&self, (emu, instrument): DynarecBlockArgs) {
        // reset delta clock before running
        emu.cpu.d_clock = 0;

        if instrument {
            (self.function.func)
                .instrument(tracing::info_span!("fn", addr = ?self.function.func))
                .inner()(&mut emu.cpu, &mut emu.mem)
        } else {
            (self.function.func)(&mut emu.cpu, &mut emu.mem)
        };

        IO::run_timer_pipeline(&mut emu.cpu, &mut emu.mem);
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

        if enabled!(Level::TRACE) {
            use std::fs::File;
            use std::io::Write;
            File::create("/tmp/jit_code.bin")?.write_all(exec.as_ref())?;
            tracing::trace!("Wrote {} bytes to /tmp/jit_code.bin", exec.len());
        }

        let func = unsafe { std::mem::transmute::<*const u8, fn(_, _)>(exec.as_ptr()) };
        Ok(DynarecFunction {
            func,
            exec: Rc::new(exec),
        })
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
        tracing::trace!("store: guest r{}", guest_reg);

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

        let offset = u32::try_from(Cpu::reg_offset(gr1)).expect("invalid cpu offset");
        tracing::trace!("store: guest ${} & ${} (pair)", reg_str(gr1), reg_str(gr2));

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

    fn emit_block_epilogue(&mut self, d_clock: u16, new_pc: u32) {
        // emit write back to dirty registers loop through every pair of
        // consecutive registers. there are 32 registers so the remainder of the
        // iterator is guaranteed to be 0
        if cfg!(debug_assertions) {
            let chunk_iter = self
                .reg_alloc
                .dirty
                .iter()
                .enumerate()
                .array_chunks::<2>()
                .into_remainder()
                .unwrap();
            debug_assert!(chunk_iter.count() == 0);
        }
        let chunk_iter = self.reg_alloc.dirty.iter().enumerate().array_chunks::<2>();
        chunk_iter
            .map(|arr| arr.map(|(guest_reg, dirty)| if *dirty { Some(guest_reg) } else { None }))
            // holy map chain
            .map(|arr| {
                arr.map(|guest_reg| {
                    guest_reg.and_then(|guest_reg| {
                        let host_reg = REG_MAP[guest_reg];
                        Some(guest_reg as u8).zip(host_reg)
                    })
                })
            })
            .for_each(|arr| match arr {
                [None, None] => {}
                [Some((guest_reg, host_reg)), None] | [None, Some((guest_reg, host_reg))] => {
                    Self::emit_writeback_free(&mut self.asm, guest_reg, host_reg);
                }
                #[cfg(target_arch = "aarch64")]
                [Some(a), Some(b)] if a.1.consecutive(b.1) => {
                    Self::emit_writeback_pair_free(&mut self.asm, [a, b]);
                }
                [Some(a), Some(b)] => {
                    Self::emit_writeback_free(&mut self.asm, a.0, a.1);
                    Self::emit_writeback_free(&mut self.asm, b.0, b.1);
                }
            });

        // emit pc update & clock update
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            const {
                debug_assert!(Cpu::D_CLOCK_OFFSET == Cpu::PC_OFFSET + 4);
            };
            dynasm!(
                self.asm
                ; .arch aarch64
                ; mov w24, new_pc as _
                ; mov w25, d_clock as _
                ; stp w24, w25, [x0, Cpu::PC_OFFSET as _]
            )
        }

        // emit return
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            dynasm!(
                self.asm
                ; .arch aarch64
                // DONE: restore all registers
                ; ldp x27, x28, [sp], #16
                ; ldp x25, x26, [sp], #16
                ; ldp x23, x24, [sp], #16
                ; ret
            )
        }
    }
    fn map_reg(&self, guest_reg: u8, spill_to: Reg) -> Result<Reg, Reg> {
        let host_reg = REG_MAP[guest_reg as usize];
        match host_reg {
            Some(host_reg) => Ok(host_reg),
            None => Err(spill_to),
        }
    }
    fn emit_load_reg(&mut self, guest_reg: u8, spill_to: Reg) -> Result<Reg, Reg> {
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::useless_conversion)]
        {
            match self.map_reg(guest_reg, spill_to) {
                Ok(Reg::W(host_reg)) => {
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
                    tracing::trace!("load: guest r{}", guest_reg);

                    Ok(Reg::W(host_reg))
                }
                // spill register
                Err(spill_to) => {
                    let offset =
                        u32::try_from(Cpu::reg_offset(guest_reg)).expect("invalid cpu offset");
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; ldr W(spill_to), [x0, offset]
                    );
                    self.reg_alloc.dirty.set(guest_reg as usize, false);
                    tracing::trace!("load: guest r{} (spill)", guest_reg);

                    Err(spill_to)
                }
            }
        }
    }

    fn writeback(&mut self, guest: u8, host: Result<Reg, Reg>) {
        let reg_inner = host.into_inner();
        if host.is_ok() {
            self.mark_dirty(guest);
        } else {
            self.emit_writeback(guest, reg_inner);
        }
    }

    fn mark_dirty(&mut self, guest_reg: u8) {
        self.reg_alloc.dirty.set(guest_reg as usize, true);
    }

    fn mark_clean(&mut self, guest_reg: u8) {
        self.reg_alloc.dirty.set(guest_reg as usize, false);
    }

    fn set_delay_slot(&mut self, emitter: impl Into<fn(EmitCtx) -> EmitSummary>) {
        let emitter = emitter.into();
        self.delay_slot = Some(emitter);
    }
}

impl Emu {
    fn linear_fetch(&self) -> impl Iterator<Item = (OpCode, DecodedOpNew)> {
        self.linear_fetch_no_decode()
            .map(|op| (op, DecodedOpNew::new(op)))
            .take_while(|(_, op)| !op.is_boundary())
    }
    fn linear_fetch_no_decode(&self) -> impl Iterator<Item = OpCode> {
        const CHUNK: usize = 32;
        (self.cpu.pc..)
            .step_by(size_of::<u32>() * CHUNK)
            .flat_map(|address| {
                self.read::<[Simd<u32, 4>; CHUNK / max_simd_elements::<u32>()], ext::NoExt>(address)
            })
            .flat_map(|value| value.to_array())
            .map(OpCode)
    }
}

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
    pub fn run_once(mut self, emu: &mut Emu) -> color_eyre::Result<Self> {
        for _ in 0..3 {
            self = self.step(emu)?;
        }
        Ok(self)
    }
    pub fn step(self, emu: &mut Emu) -> color_eyre::Result<Self> {
        match self {
            PipelineV2::Init { pc, dynarec } => match emu.dynarec_cache.take(pc) {
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
            PipelineV2::Compiled { pc, func, dynarec } => {
                func(emu, false);
                Ok(PipelineV2::Called { pc, func, dynarec })
            }
            PipelineV2::Called { pc, func, dynarec } => {
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
    let ops = emu
        .linear_fetch_no_decode()
        .take(op_count)
        .map(|op| op.0)
        .collect::<Vec<_>>();
    let mut hasher = rapidhash::fast::RapidHasher::default();
    ops.hash(&mut hasher);

    hasher.finish()
}

fn fetch_and_compile_single_threaded(
    emu: &Emu,
    mut dynarec: Box<Dynarec>,
) -> color_eyre::Result<DynarecBlock> {
    dynarec.emit_block_prelude(&emu.cpu, &emu.mem);

    let mut hasher = rapidhash::fast::RapidHasher::default();
    let mut op_count = 0;
    let cycles = emu.linear_fetch().fold(0u16, |cycles, (op, decoded)| {
        decoded.emit(EmitCtx {
            dynarec: &mut dynarec,
        });
        hasher.write_u32(op.0);
        op_count += 1;
        cycles + decoded.cycles()
    });
    let new_pc = emu.cpu.pc + op_count as u32 * size_of::<OpCode>() as u32;
    dynarec.emit_block_epilogue(cycles, new_pc);

    let func = dynarec.finalize()?;
    let hash = hasher.finish();

    Ok(DynarecBlock {
        function: func,
        op_count,
        hash,
    })
}

#[cfg(test)]
mod tests {
    use std::{ptr, time::Instant};
    extern crate test;

    use color_eyre::Result;
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{
        Emu,
        cpu::{ops::addiu::*, program},
    };

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
        dynarec.emit_block_epilogue(0, 0x0);
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
        dynarec.emit_block_epilogue(0, 0x0);
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

    #[test]
    fn dynarec_v2_test_updates() -> Result<()> {
        setup_tracing();
        let mut emu = Emu::default();
        emu.write_many(
            0x0,
            &program([addiu(10, 0, 4), addiu(10, 0, 5), OpCode(69420)]),
        );
        PipelineV2::new(&emu).run_once(&mut emu)?;
        tracing::info!(?emu.cpu);

        assert_eq!(emu.cpu.gpr[10], 5);
        assert_eq!(emu.cpu.d_clock, 2);
        assert_eq!(emu.cpu.pc, 0x8);
        Ok(())
    }

    #[bench]
    fn dynarec_v2_test_50_adds(b: &mut test::Bencher) -> Result<()> {
        let instruction_count = 50;
        setup_tracing();
        let mut emu = Emu::default();
        let mut pipe = PipelineV2::new(&emu);
        let mut last_addr = 0x0;
        for addr in (0x0..).step_by(4).take(instruction_count) {
            emu.write(addr, addiu(8, 8, 1));
            last_addr = addr;
        }
        emu.write(last_addr + 4, OpCode::HALT_FIELDS);

        emu.cpu.gpr[8] = 0;
        for i in 0..3 {
            let now = Instant::now();
            pipe = pipe.step(&mut emu)?;
            let elapsed = now.elapsed().as_micros();
            tracing::info!("{i}: elapsed: {}us", elapsed);
        }

        assert_eq!(emu.cpu.gpr[8], instruction_count as u32);

        tracing::info!("after cache:");

        for j in 0..3 {
            emu.cpu.pc = 0x0;
            for i in 0..3 {
                let now = Instant::now();
                pipe = pipe.step(&mut emu)?;
                let elapsed = now.elapsed().as_micros();
                tracing::info!("{i}: elapsed: {}us", elapsed);
            }
            println!();
        }

        Ok(())
    }

    #[bench]
    fn dynarec_v2_test_50_adds_call(b: &mut test::Bencher) -> Result<()> {
        let instruction_count = 50;
        setup_tracing();
        let mut emu = Emu::default();
        let mut pipe = PipelineV2::new(&emu);
        let mut last_addr = 0x0;
        for addr in (0x0..).step_by(4).take(instruction_count) {
            emu.write(addr, addiu(8, 8, 1));
            last_addr = addr;
        }
        emu.write(last_addr + 4, OpCode::HALT_FIELDS);

        emu.cpu.gpr[8] = 0;
        pipe = pipe.step(&mut emu)?;
        let PipelineV2::Compiled { pc, func, dynarec } = pipe else {
            panic!("wrong stage");
        };

        b.iter(|| {
            func(&mut emu, false);
        });

        Ok(())
    }
}
