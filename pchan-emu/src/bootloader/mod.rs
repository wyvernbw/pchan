use std::{fs, io::Read, path::PathBuf};

use thiserror::Error;

use crate::memory::{Memory, buffer, from_kb, kb};

#[derive(derive_more::Debug)]
pub struct Bootloader {
    bios_path: PathBuf,
}

impl Default for Bootloader {
    fn default() -> Self {
        let bios_path = std::env::var("PCHAN_BIOS").unwrap_or("./SCPH1001.BIN".to_owned());
        Bootloader {
            bios_path: bios_path.into(),
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
        let mut bios = buffer(kb(524));
        let _ = bios_file
            .read(&mut bios)
            .map_err(BootError::BiosReadError)?;
        let bios_slice = &bios[..kb(512)];
        tracing::info!("loaded bios: {}kb", from_kb(bios_slice.len()));

        mem.write_many(0xBFC0_0000, bios_slice);

        Ok(())
    }
}
