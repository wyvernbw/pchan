use crate::{Bus, Emu, io::UnhandledIO};

pub struct CDRomState {}

pub trait CDRom: Bus {
    fn write<T>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        // TODO
        Err(UnhandledIO(address))
    }
    fn read<T>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        // TODO
        Err(UnhandledIO(address))
    }
}

impl CDRom for Emu {}
