use byteorder::{LE, ReadBytesExt};
use pchan_utils::hex;
use std::{
    borrow::Cow,
    fs,
    io::{BufRead, Cursor, Read},
    marker::PhantomData,
    path::{Path, PathBuf},
    string::FromUtf8Error,
};
use thiserror::Error;
use tracing::instrument;

use crate::{
    Bus, cpu,
    io::IO,
    memory::{self, GUEST_MEM_MAP, MEM_MAP},
};
use crate::{
    Emu,
    memory::{buffer, from_kb, kb},
};

#[cfg(feature = "amidog-tests")]
pub static AMIDOG_TESTS: &[u8] =
    include_bytes!("../../tests/assets/amidog_psxtest_cpu/psxtest_cpu.exe");

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

    #[error("could not sideload exe: {0}")]
    SideloadErr(#[from] ExeHeaderParseErr),
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

    fn run_sideloading(&mut self, exe: &[u8]) -> Result<(), BootError> {
        match self.cpu().pc {
            0x80030000 => self.sideload_exe(exe),
            _ => Ok(()),
        }
    }

    #[instrument(err, skip_all)]
    fn sideload_exe(&mut self, exe: &[u8]) -> Result<(), BootError> {
        let exe = Exe::parse(exe)?.to_owned_code();
        tracing::info!("parsed executable");
        self.cpu_mut().pc = exe.header.initial_pc;
        tracing::info!("header = {:#?}", exe.header);
        self.cpu_mut().gpr[cpu::GP as usize] = exe.header.initial_gp;
        if exe.header.sp_fp_base != 0 {
            self.cpu_mut().gpr[cpu::SP as usize] = exe.header.sp_fp_base;
            self.cpu_mut().gpr[cpu::FP as usize] = exe.header.sp_fp_base;
        }

        // copy code to memory
        exe.code
            .iter()
            .copied()
            .take(exe.header.filesize as usize * kb(2))
            .enumerate()
            .for_each(|(idx, byte)| {
                let address = idx + exe.header.dest_addr as usize;
                self.write::<u8>(address as u32, byte);
            });

        tracing::info!("set state");
        Ok(())
    }
}

impl Bootloader for Emu {}

/// ```md
///  000h-007h ASCII ID "PS-X EXE"
///  008h-00Fh Zerofilled
///  010h      Initial PC                   (usually 80010000h, or higher)
///  014h      Initial GP/R28               (usually 0)
///  018h      Destination Address in RAM   (usually 80010000h, or higher)
///  01Ch      Filesize (must be N*800h)    (excluding 800h-byte header)
///  020h      Data section Start Address   (usually 0)
///  024h      Data Section Size in bytes   (usually 0)
///  028h      BSS section Start Address    (usually 0) (when below Size=None)
///  02Ch      BSS section Size in bytes    (usually 0) (0=None)
///  030h      Initial SP/R29 & FP/R30 Base (usually 801FFFF0h) (or 0=None)
///  034h      Initial SP/R29 & FP/R30 Offs (usually 0, added to above Base)
///  038h-04Bh Reserved for A(43h) Function (should be zerofilled in exefile)
///  04Ch-xxxh ASCII marker
///             "Sony Computer Entertainment Inc. for Japan area"
///             "Sony Computer Entertainment Inc. for Europe area"
///             "Sony Computer Entertainment Inc. for North America area"
///             (or often zerofilled in some homebrew files)
///             (the BIOS doesn't verify this string, and boots fine without it)
///  xxxh-7FFh Zerofilled
///  800h...   Code/Data                  (loaded to entry[018h] and up)
/// ```
#[derive(derive_more::Debug, Clone)]
#[repr(C)]
pub struct ExeHeader {
    #[debug("{}", bstr::BStr::new(&self.ascii_id))]
    ascii_id:     [u8; 8],
    #[debug("{}", hex(self.initial_pc))]
    initial_pc:   u32,
    #[debug("{}", hex(self.initial_gp))]
    initial_gp:   u32,
    #[debug("{}", hex(self.dest_addr))]
    dest_addr:    u32,
    filesize:     u32,
    #[debug("{}", hex(self.data_addr))]
    data_addr:    u32,
    #[debug("{}", hex(self.data_size))]
    data_size:    u32,
    #[debug(skip)]
    bss_start:    u32,
    #[debug(skip)]
    bss_size:     u32,
    #[debug("{}", hex(self.sp_fp_base))]
    sp_fp_base:   u32,
    #[debug("{}", hex(self.sp_fp_offset))]
    sp_fp_offset: u32,
    ascii_marker: String,
}

#[derive(Debug, Error)]
pub enum ExeHeaderParseErr {
    #[error(transparent)]
    IoErr(#[from] std::io::Error),
    #[error(transparent)]
    Utf8Err(#[from] FromUtf8Error),
}

impl ExeHeader {
    pub fn parse(from: &[u8]) -> Result<Self, ExeHeaderParseErr> {
        let mut rdr = Cursor::new(from);
        // FIXME: this is wrong for big endian woops
        let ascii_id = rdr.read_array::<8>()?;

        rdr.set_position(0x10);

        let initial_pc = rdr.read_u32::<LE>()?;
        let initial_gp = rdr.read_u32::<LE>()?;
        let dest_addr = rdr.read_u32::<LE>()?;

        let filesize = rdr.read_u32::<LE>()?;

        let data_addr = rdr.read_u32::<LE>()?;
        let data_size = rdr.read_u32::<LE>()?;
        let bss_start = rdr.read_u32::<LE>()?;
        let bss_size = rdr.read_u32::<LE>()?;
        let sp_fp_base = rdr.read_u32::<LE>()?;
        let sp_fp_offset = rdr.read_u32::<LE>()?;

        rdr.set_position(0x4c);

        // read null terminated string
        let mut marker = Vec::with_capacity(800);
        rdr.read_until(b'\0', &mut marker)?;
        if cfg!(target_endian = "big") {
            marker.reverse();
        }
        let ascii_marker = String::from_utf8(marker)?;

        Ok(Self {
            ascii_id,
            initial_pc,
            initial_gp,
            dest_addr,
            filesize,
            data_addr,
            data_size,
            bss_start,
            bss_size,
            sp_fp_base,
            sp_fp_offset,
            ascii_marker,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Exe<'a, B> {
    header: ExeHeader,
    code:   B,
    _life:  PhantomData<&'a ()>,
}

impl<'a> Exe<'a, Cow<'a, [u8]>> {
    fn parse(from: &'a [u8]) -> Result<Self, ExeHeaderParseErr> {
        let header_bytes = &from[0..kb(2)];
        let header = ExeHeader::parse(header_bytes)?;
        let code = &from[kb(2)..];
        if cfg!(target_endian = "big") {
            todo!("handle big endian executable parsing");
        }

        Ok(Self {
            header,
            code: Cow::Borrowed(code),
            _life: PhantomData,
        })
    }

    fn to_owned_code(&self) -> Exe<'static, Vec<u8>> {
        Exe {
            header: self.header.clone(),
            code:   self.code.to_vec(),
            _life:  PhantomData,
        }
    }
}
