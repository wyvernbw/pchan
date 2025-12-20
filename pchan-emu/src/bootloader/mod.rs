use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::{
    Bus,
    io::IO,
    memory::{GUEST_MEM_MAP, MEM_MAP},
};
use crate::{
    Emu,
    memory::{buffer, from_kb, kb},
};

#[derive(derive_more::Debug, Clone)]
pub struct BootloaderState {
    bios_path: PathBuf,
}

impl Default for BootloaderState {
    fn default() -> Self {
        let bios_path = std::env::var("PCHAN_BIOS").unwrap_or("./SCPH1001.BIN".to_owned());
        BootloaderState {
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

pub trait Bootloader: Bus + IO {
    fn set_bios_path(&mut self, path: impl AsRef<Path>) {
        self.bootloader_mut().bios_path = path.as_ref().to_path_buf();
    }
    fn load_bios(&mut self) -> Result<(), BootError> {
        let mut bios_file =
            fs::File::open(&self.bootloader().bios_path).map_err(BootError::BiosFileOpenError)?;
        let mut bios = buffer(kb(524));
        let _ = bios_file
            .read(&mut bios)
            .map_err(BootError::BiosReadError)?;
        let bios_slice = &bios[..kb(512)];

        // NOTE: we cannot use the typical IO interace here because bios
        // is technically not writeable
        {
            let this = &mut *self;
            let mut address = 0xBFC0_0000;
            for value in bios_slice.iter().copied() {
                this.mem_mut()
                    .write_region(MEM_MAP.bios, GUEST_MEM_MAP.bios, address, value);
                address += 0x1;
            }
        };
        tracing::info!("loaded bios: {}kb", from_kb(bios_slice.len()));

        Ok(())
    }
}

impl Bootloader for Emu {}
