#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]
#![feature(const_from)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(slice_as_array)]
#![feature(portable_simd)]
#![feature(iter_collect_into)]

use std::error::Error;

use crate::{bootloader::Bootloader, cpu::Cpu, memory::Memory};

pub mod bootloader;
pub mod cpu;
pub mod memory;

#[derive(Default)]
pub struct Emu {
    cpu: Cpu,
    mem: Memory,
    boot: Bootloader,
}

impl Emu {
    fn run(mut self) -> Result<(), Box<dyn Error + 'static>> {
        self.boot.load_bios(&mut self.mem)?;
        loop {
            let interrupt = self.cpu.run_cycle(&mut self.mem);
            self.cpu.advance_cycle();
            match interrupt {
                None => {}
                Some(_) => {}
            }
        }
    }
}
