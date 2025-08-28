use std::{fs, io::Read, path::PathBuf};

use thiserror::Error;

use crate::memory::{KSEG1Addr, Memory, buffer, kb};

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
    pub(crate) fn load_bios(&self, mem: &mut Memory) -> Result<(), BootError> {
        let mut bios_file =
            fs::File::open(&self.bios_path).map_err(BootError::BiosFileOpenError)?;
        let mut bios = buffer(kb(600));
        let n = bios_file
            .read(&mut bios)
            .map_err(BootError::BiosReadError)?;

        for (offset, &byte) in bios[..n].iter().enumerate() {
            let offset = offset as u32;
            // load bios with KSEG0 for caching and better performance (i think)
            // note that the cache is not implemented yet
            mem.write(KSEG1Addr(0xBFC00000 + offset), byte);
        }
        Ok(())
    }
}
