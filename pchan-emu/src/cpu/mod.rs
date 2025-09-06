use std::fmt::Display;

#[cfg(test)]
mod cranelift_tests;
pub mod ops;

#[derive(Default)]
#[repr(C)]
pub(crate) struct Cpu {
    pub(crate) gpr: [u64; 32],
    pub(crate) pc: u64,
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

type Reg = usize;

const RA: Reg = 31;
