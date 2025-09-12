use std::fmt::Display;

use pchan_utils::array;

#[cfg(test)]
mod cranelift_tests;
pub mod ops;

#[derive(Default, derive_more::Debug)]
#[repr(C)]
pub struct Cpu {
    #[debug("{}",
        gpr
            .iter()
            .enumerate()
            .filter(|(_, x)| x != &&0)
            .map(|(i, x)| format!("${}={}", REG_STR[i], x))
            .intersperse(" ".to_string())
            .collect::<String>()
    )]
    pub gpr: [u32; 32],
    pub pc: u32,
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

impl Cpu {
    pub fn clear_registers(&mut self) {
        self.gpr = [0u32; 32];
    }
}

type Reg = usize;

const RA: Reg = 31;

pub const REG_STR: &[&str] = &array![
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
