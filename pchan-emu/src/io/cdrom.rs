use crate::{Bus, Emu};

pub struct CDRomState {}

pub trait CDRom: Bus {
    fn write<T>(&mut self, address: u32, value: T) -> Result<(), ()> {
        todo!()
    }
    fn read<T>(&self, address: u32) -> Result<T, ()> {
        todo!()
    }
}

impl CDRom for Emu {}
