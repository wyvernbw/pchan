use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
};

use thiserror::Error;

use crate::memory::{Memory, buffer, kb};

pub struct Bootloader {
    bios_path: PathBuf,
}

impl Default for Bootloader {
    fn default() -> Self {
        Bootloader {
            bios_path: "./SCPH1001.BIN".into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum BootError {
    #[error(transparent)]
    BiosFileOpenError(std::io::Error),
    #[error("bios file could not be read: {0}")]
    BiosReadError(std::io::Error),
}

impl Bootloader {
    pub fn load_bios(&self, mem: &mut Memory) -> Result<(), BootError> {
        let mut bios_file =
            fs::File::open(&self.bios_path).map_err(BootError::BiosFileOpenError)?;
        let mut bios = buffer(kb(600));
        let _ = bios_file
            .read(&mut bios)
            .map_err(BootError::BiosReadError)?;

        mem.as_mut()
            .write_all(&bios)
            .map_err(BootError::BiosReadError)?;

        Ok(())
    }
}
