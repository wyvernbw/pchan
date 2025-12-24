use std::{fmt::Display, mem::offset_of};

use bitfield::bitfield;
use pchan_utils::{array, hex};
use tracing::instrument;

use crate::cpu::ops::OpCode;

pub mod ops;

#[derive(Default, derive_more::Debug, Clone, Hash)]
#[repr(C)]
pub struct Cpu {
    #[debug("{:#?}",
        gpr
            .iter()
            .enumerate()
            .filter(|(_, x)| x != &&0)
            .map(|(i, x)| format!("${}={}", REG_STR[i], hex(*x)))
            .collect::<Vec<String>>()
    )]
    pub gpr:     [u32; 32],
    #[debug("{}", hex(self.pc))]
    pub pc:      u32, // store pc and d_clock together so one write can target both
    pub d_clock: u32,
    pub hilo:    u64,
    pub cop0:    Cop0,
    pub cop1:    Cop1,
    pub cop2:    Cop2,
}

macro_rules! coprocessor_definition {
    ($n:ident) => {
        #[derive(derive_more::Debug, Clone, Hash)]
        #[repr(C)]
        pub struct $n {
            #[debug("{:#?}", reg.iter() .enumerate() .filter(|(_, x)| x != &&0) .map(|(i, x)|format!("${}={}", REG_STR[i], hex(*x))) .collect::<Vec<String>>())]
            pub reg: [u32; 32],
        }
    };
}

coprocessor_definition!(Cop0);
coprocessor_definition!(Cop1);
coprocessor_definition!(Cop2);

bitfield! {
    pub struct Cop0StatusReg(u32);

    // TODO: other fields
    bev, set_bev: 22;
    isc, set_isc: 16;
}

impl Default for Cop0 {
    fn default() -> Self {
        let mut reg = [0u32; 32];

        let mut r12 = Cop0StatusReg(0);
        r12.set_bev(true);
        reg[12] = r12.0;

        Self { reg }
    }
}

impl Cop0 {
    pub fn bev(&self) -> bool {
        self.reg[12] & (1 << 22) != 0
    }
    pub fn isc(&self) -> bool {
        Cop0StatusReg(self.reg[12]).isc()
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
        self.gpr = [0u32; 32];
    }
    pub fn jump_to_bios(&mut self) {
        self.pc = 0xBFC0_0000;
    }

    pub fn isc(&self) -> bool {
        self.cop0.isc()
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Cop1 {
    fn default() -> Self {
        Self { reg: [0; 32] }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Cop2 {
    fn default() -> Self {
        Self { reg: [0; 32] }
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
    31 => "ra"
];

pub const fn reg_str(reg: Reg) -> &'static str {
    REG_STR[reg as usize]
}

pub fn program<const N: usize>(prog: [OpCode; N]) -> [u32; N] {
    prog.map(|op| op.0)
}
