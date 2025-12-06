use std::fmt::Display;

use bitfield::bitfield;
use pchan_utils::{array, hex};
use tracing::instrument;

use crate::cpu::ops::OpCode;

pub mod ops;

#[derive(Default, derive_more::Debug)]
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
    pub gpr: [u32; 32],
    #[debug("{}", hex(self.pc))]
    pub pc: u32,
    pub hilo: u64,
    pub d_clock: u16,
    pub cop0: Cop0,
    pub cop1: Cop1,
    #[debug(skip)]
    pub _pad_cop2_gte: [u64; 32],
}

macro_rules! coprocessor_definition {
    ($n:ident) => {
        #[derive(derive_more::Debug)]
        #[repr(C)]
        pub struct $n {
            #[debug("{:#?}", reg.iter() .enumerate() .filter(|(_, x)| x != &&0) .map(|(i, x)|format!("${}={}", REG_STR[i], hex(*x))) .collect::<Vec<String>>())]
            pub reg: [u32; 64],
        }
    };
}

coprocessor_definition!(Cop0);
coprocessor_definition!(Cop1);

bitfield! {
    pub struct Cop0StatusReg(u32);

    // TODO: other fields
    bev, set_bev: 22;
    isc, set_isc: 16;
}

impl Default for Cop0 {
    fn default() -> Self {
        let mut reg = [0u32; 64];

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

bitfield! {
    #[derive(Clone, Copy)]
    pub struct CauseRegister(u32);

    excode, set_excode: 6, 2;
    interrupt_pending, set_interrupt_pending: 15, 8;
    cop_number, set_cop_number: 29, 28;
    branch_delay, set_branch_delay: 31;
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Exception {
    Interrupt = 0x0,
    Break = 0x9,
}

impl Cpu {
    #[instrument(ret)]
    pub fn handle_exception(&mut self, exception: Exception) {
        let cause = self.cop0.reg[13];
        let mut cause = CauseRegister(cause);
        cause.set_excode(exception as u32);
        self.cop0.reg[13] = cause.0;

        self.cop0.reg[14] = self.pc;

        self.pc = match self.cop0.bev() {
            false => 0x8000_0080,
            true => 0xbfc0_0180,
        }
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

    #[unsafe(no_mangle)]
    pub fn handle_rfe(&mut self) {
        tracing::info!("running rfe");
        let sr = self.cop0.reg[12];
        self.cop0.reg[12] = (sr & !0x3F) | ((sr >> 2) & 0x3F);
        // panic!("rfe breakpoint");
    }

    #[unsafe(no_mangle)]
    pub fn handle_break(&mut self) {
        tracing::info!("running break");
        self.handle_exception(Exception::Break);
    }
}

impl Default for Cop1 {
    fn default() -> Self {
        Self { reg: [0; 64] }
    }
}

pub type Reg = u8;

const RA: Reg = 31;
const SP: Reg = 29;

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
