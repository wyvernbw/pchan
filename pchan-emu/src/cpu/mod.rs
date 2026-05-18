use std::{
    fmt::Display,
    mem::offset_of,
    ops::{Index, IndexMut},
};

use bitbybit::{bitenum, bitfield};
use derive_more as d;
use pchan_utils::{array, hex};

use crate::{
    cpu::{exceptions::CauseRegister, ops::OpCode},
    io::irq::IrqState,
};

pub mod ops;

#[derive(Default, derive_more::Debug, Clone, Hash)]
#[repr(C)]
pub struct Cpu {
    pub gpr:          Regs<32, true>,
    #[debug("{}", hex(self.pc))]
    pub pc:           u32, // store pc and d_clock together so one write can target both
    pub d_clock:      u32,
    pub hilo:         u64,
    pub scratch_buf:  [u32; Self::SCRATCH_SIZE],
    pub cop0:         Cop0,
    pub cop2:         Cop2,
    pub cop1:         Cop1,
    pub vblank_timer: u32,
    pub cycles:       u64,
    pub irq:          IrqState,
    pub jump_queue:   Option<u32>,
}

use std::fmt;

#[derive(d::Deref, d::DerefMut, d::AsMut, d::AsRef, Hash, Clone)]
pub struct Regs<const N: usize, const NAMED: bool>([u32; N]);

impl<const N: usize, const NAMED: bool> Default for Regs<N, NAMED> {
    fn default() -> Self {
        let arr = [0; N];
        Self(arr)
    }
}

impl<const N: usize, const NAMED: bool> fmt::Debug for Regs<N, NAMED> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (i, val) in self
            .0
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, val)| val != &0)
        {
            if NAMED {
                list.entry(&format_args!("${}={}", REG_STR[i], hex(val)));
            } else {
                list.entry(&format_args!("$r{}={}", i, hex(val)));
            }
        }
        list.finish()
    }
}

macro_rules! coprocessor_definition {
    ($n:ident, $cnt:expr) => {
        #[derive(derive_more::Debug, Clone, Hash)]
        #[repr(C)]
        pub struct $n {
            pub reg: Regs<$cnt, true>,
        }
    };
    ($n:ident, $cnt:expr, unnamed) => {
        #[derive(derive_more::Debug, Clone, Hash)]
        #[repr(C)]
        pub struct $n {
            pub reg: Regs<$cnt, false>,
        }
    };
}

coprocessor_definition!(Cop0, 32);
coprocessor_definition!(Cop1, 32);
coprocessor_definition!(Cop2, 64, unnamed);

// bitfield! {
//     pub struct Cop0StatusReg(u32);

//     // TODO: other fields
//     bev, set_bev: 22;
//     isc, set_isc: 16;
// }

#[bitfield(u32, debug)]
pub struct Cop0StatusReg {
    /// interrupt enable current
    #[bit(0, rw)]
    iec: bool,
    /// kernel/user current
    #[bit(1, rw)]
    kuc: KernelUserMode,
    /// interrupt enable previous
    #[bit(2, rw)]
    iep: bool,
    /// kernel/user previous
    #[bit(3, rw)]
    kup: KernelUserMode,
    /// interrupt enable old
    #[bit(4, rw)]
    ieo: bool,
    /// kernel/user old
    #[bit(5, rw)]
    kuo: KernelUserMode,

    #[bit(8, rw)]
    irq_mask: [bool; 8],

    #[bits(8..=15, rw)]
    irq_mask_combined: u8,

    #[bit(16, rw)]
    isc: bool,
    #[bit(22, rw)]
    bev: bool,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum KernelUserMode {
    Kernel = 0x0,
    User   = 0x1,
}

impl Default for Cop0 {
    fn default() -> Self {
        let mut reg = Regs([0u32; 32]);

        let mut r12 = Cop0StatusReg::new_with_raw_value(0);
        r12.set_bev(true);
        reg[12] = r12.raw_value();

        Self { reg }
    }
}

impl Cop0 {
    pub fn status(&self) -> Cop0StatusReg {
        Cop0StatusReg::new_with_raw_value(self.reg[12])
    }
    pub fn cause(&self) -> CauseRegister {
        CauseRegister::new_with_raw_value(self.reg[13])
    }
    pub fn set_cause(&mut self, cause: CauseRegister) {
        self.reg[13] = cause.raw_value();
    }
    pub fn update_cause(&mut self, f: impl FnOnce(CauseRegister) -> CauseRegister) {
        let cause = self.cause();
        let new_cause = f(cause);
        self.set_cause(new_cause);
    }
    pub fn set_bd(&mut self, value: bool) {
        self.reg[13] = self.cause().with_bd(value).raw_value()
    }
    pub fn set_bt(&mut self, value: bool) {
        self.reg[12] = self.cause().with_bt(value).raw_value();
    }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn emu_set_bd(emu: &mut crate::Emu) {
    emu.cpu.cop0.set_bd(true);
}
#[inline(never)]
#[unsafe(no_mangle)]
pub fn emu_set_bt_true(emu: &mut crate::Emu) {
    emu.cpu.cop0.set_bt(true);
}
#[inline(never)]
#[unsafe(no_mangle)]
pub fn emu_set_bt_false(emu: &mut crate::Emu) {
    emu.cpu.cop0.set_bt(false);
}

impl Cop0StatusReg {
    pub fn push_exception_stack(&mut self) {
        self.set_kuo(self.kup());
        self.set_ieo(self.iep());
        self.set_kup(self.kuc());
        self.set_iep(self.iec());
        self.set_kuo(KernelUserMode::Kernel);
        self.set_iec(false);
    }
}

impl Display for Cpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpr = self
            .gpr
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(idx, value)| format!("{idx}={value}"))
            .intersperse(",".to_string())
            .collect::<String>();
        let gpr = if gpr.is_empty() {
            "None".to_string()
        } else {
            gpr
        };
        write!(f, "cpu:gpr[{gpr}]")
    }
}

pub mod exceptions;

impl Cpu {
    pub const PC_OFFSET: usize = offset_of!(Self, pc);
    pub const HILO_OFFSET: usize = offset_of!(Self, hilo);
    pub const D_CLOCK_OFFSET: usize = offset_of!(Self, d_clock);
    pub const SCRATCH_OFFSET: usize = offset_of!(Cpu, scratch_buf);
    pub const SCRATCH_SIZE: usize = 8;

    pub const fn reg_offset(reg: u8) -> usize {
        (offset_of!(Cpu, gpr) + size_of::<u32>() * reg as usize)
    }

    pub const fn cop_reg_offset(cop: u8, reg: u8) -> usize {
        match cop {
            0 => Self::cop0_reg_offset(reg),
            1 => Self::cop1_reg_offset(reg),
            2 => Self::cop2_reg_offset(reg),
            _ => todo!(),
        }
    }

    pub const fn cop0_reg_offset(reg: u8) -> usize {
        offset_of!(Cpu, cop0) + offset_of!(Cop0, reg) + size_of::<u32>() * reg as usize
    }

    pub const fn cop1_reg_offset(reg: u8) -> usize {
        offset_of!(Cpu, cop1) + offset_of!(Cop1, reg) + size_of::<u32>() * reg as usize
    }

    pub const fn cop2_reg_offset(reg: u8) -> usize {
        offset_of!(Cpu, cop2) + offset_of!(Cop2, reg) + size_of::<u32>() * reg as usize
    }

    pub fn clear_registers(&mut self) {
        self.gpr = Regs([0u32; 32]);
    }
    pub fn jump_to_bios(&mut self) {
        self.pc = 0xBFC0_0000;
    }

    pub fn isc(&self) -> bool {
        self.cop0.status().isc()
    }

    pub fn enqueue_jump(&mut self, address: u32) {
        self.jump_queue = Some(address);
    }

    pub fn drain_jump_queue(&mut self) {
        if let Some(address) = self.jump_queue.take() {
            self.pc = address;
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Cop1 {
    fn default() -> Self {
        Self { reg: Regs([0; 32]) }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Cop2 {
    fn default() -> Self {
        Self { reg: Regs([0; 64]) }
    }
}

pub type Reg = u8;

pub(crate) const GP: Reg = 28;
pub(crate) const SP: Reg = 29;
pub(crate) const FP: Reg = 30;
pub(crate) const RA: Reg = 31;

pub static REG_STR: &[&str] = &array![
     0 => "zero",
     1 => "at",
     2 => "v0",
     3 => "v1",
     4 => "a0",
     5 => "a1",
     6 => "a2",
     7 => "a3",
     8 => "t0",
     9 => "t1",
    10 => "t2",
    11 => "t3",
    12 => "t4",
    13 => "t5",
    14 => "t6",
    15 => "t7",
    16 => "s0",
    17 => "s1",
    18 => "s2",
    19 => "s3",
    20 => "s4",
    21 => "s5",
    22 => "s6",
    23 => "s7",
    24 => "t8",
    25 => "t9",
    26 => "k0",
    27 => "k1",
    28 => "gp",
    29 => "sp",
    30 => "fp(s8)",
    31 => "ra",
    32 => "hi",
    33 => "lo",
];

pub const fn reg_str(reg: Reg) -> &'static str {
    REG_STR[reg as usize]
}

pub const fn program<const N: usize>(prog: [OpCode; N]) -> [u32; N] {
    const fn raw_value(op: OpCode) -> u32 {
        op.raw_value()
    }
    prog.map(raw_value)
}

impl Index<&'static str> for Cpu {
    type Output = u32;

    fn index(&self, index: &'static str) -> &Self::Output {
        match index {
            "$sp" => &self.gpr[29],
            _ => panic!("unknown register {index}"),
        }
    }
}

impl IndexMut<&'static str> for Cpu {
    fn index_mut(&mut self, index: &'static str) -> &mut Self::Output {
        match index {
            "$sp" => &mut self.gpr[29],
            _ => panic!("unknown register {index}"),
        }
    }
}
