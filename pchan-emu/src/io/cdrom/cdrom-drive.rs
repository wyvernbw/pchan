use std::io::{BufReader, Read, Seek};

use heapless::Deque;
use smallvec::smallvec;

use crate::cpu::Cpu;
use crate::io::cdrom::cdrom_cmds::{Response, SetMode, StatusCode};
use crate::io::cdrom::cdrom_format::{CdromCursor, CueFormat, Mss};
use crate::io::cdrom::{CDRomStatusReg, CdromScheduler, DriveStatus};

#[derive(Default, derive_more::Debug)]
pub struct CdromDrive {
    pub cursor:       CdromCursor,
    pub status_code:  StatusCode,
    pub drive_status: DriveStatus,
    pub mode:         SetMode,
    drive_state:      DriveState,
    disc:             Option<Disc>,
    host_disc_err:    Option<std::io::Error>,
}

impl Clone for CdromDrive {
    fn clone(&self) -> Self {
        Self {
            cursor:        self.cursor,
            status_code:   self.status_code,
            drive_status:  self.drive_status.clone(),
            mode:          self.mode,
            drive_state:   self.drive_state.clone(),
            disc:          None,
            host_disc_err: None,
        }
    }
}

#[derive(Default, derive_more::Debug, Clone)]
enum DriveState {
    #[default]
    Idle,
    ReadN,
}

const CYCLES_PER_BYTE: u64 = Cpu::CLOCK as u64 / (2048 * 75);

impl CdromDrive {
    pub fn setloc<T>(&mut self, mss: Mss<T>)
    where
        u8: const From<T>,
    {
        self.cursor = CdromCursor::from_mss(mss);
        if let Some(disc) = &mut self.disc {
            let res = disc.seek(self.cursor);
            self.host_disc_err = res.err();
        }
    }

    pub fn setmode(&mut self, setmode: SetMode) {
        if !setmode.ignore_bit() {
            self.mode.set_sect_size(setmode.sect_size());
        }
    }

    pub fn readn(&mut self) {
        self.drive_state = DriveState::ReadN;
    }

    pub fn run(&mut self, scheduler: &mut CdromScheduler<'_>) {
        match self.drive_state {
            DriveState::Idle => {}
            DriveState::ReadN => {
                scheduler.schedule(
                    100,
                    Response::new(
                        super::HInt::Int1DataReady,
                        smallvec![self.status_code.raw_value()],
                    ),
                );
                scheduler.evque.schedule(
                    |emu, _| {
                        emu.cdrom
                            .drive
                            .request_data(&mut emu.cdrom.status, &mut emu.cdrom.data_fifo);
                        emu.cdrom.drive.run(&mut CdromScheduler {
                            evque:     &mut emu.evque,
                            responses: &mut emu.cdrom.responses,
                        });
                    },
                    0,
                    CYCLES_PER_BYTE,
                );
            }
        }
    }

    pub fn request_data(&mut self, status: &mut CDRomStatusReg, result_fifo: &mut Deque<u8, 16>) {
        status.set_data_req(true);
        if let Some(disc) = &mut self.disc {
            let (n, bytes) = match disc.readn::<1>(self.mode) {
                Ok(res) => res,
                Err(err) => {
                    self.host_disc_err = Some(err);
                    return;
                }
            };
            if n == 0 {
                return;
            }
            let res = result_fifo.push_back(bytes[0]);
            if res.is_err() {
                tracing::warn!("cdrom byte dropped");
            }
        }
    }
}

#[derive(derive_more::Debug)]
enum DiscReader {
    Streamed(StreamedDiskReader),
    InMemory(InMemoryDiskReader),
}

#[derive(Default, derive_more::Debug, Clone)]
pub struct InMemoryDiskReader {
    #[debug(skip)]
    buf:    Box<[u8]>,
    cursor: CdromCursor,
}

trait DiscFile: Read + Seek {}
impl<T> DiscFile for T where T: Read + Seek {}

#[derive(derive_more::Debug)]
struct StreamedDiskReader {
    #[debug(skip)]
    reader: BufReader<Box<dyn DiscFile>>,
    cursor: CdromCursor,
}

impl DiscReader {
    fn update_cursor(&mut self, cursor: CdromCursor) {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => {
                streamed_disk_reader.cursor = cursor;
            }
            DiscReader::InMemory(in_memory_disk_reader) => {
                in_memory_disk_reader.cursor = cursor;
            }
        }
    }
    pub fn seek(&mut self, to: u64) -> std::io::Result<()> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.seek(to),
            DiscReader::InMemory(in_memory_disk_reader) => todo!(),
        }
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        mode: SetMode,
    ) -> std::io::Result<(usize, [u8; BYTES])> {
        match self {
            DiscReader::Streamed(streamed_disk_reader) => streamed_disk_reader.readn(mode),
            DiscReader::InMemory(in_memory_disk_reader) => todo!(),
        }
    }
}

impl StreamedDiskReader {
    fn seek(&mut self, to: u64) -> std::io::Result<()> {
        self.reader.seek(std::io::SeekFrom::Start(to))?;
        Ok(())
    }

    fn readn<const BYTES: usize>(
        &mut self,
        mode: SetMode,
    ) -> std::io::Result<(usize, [u8; BYTES])> {
        let idx = self.cursor.lba_to_bytes() + self.cursor.byte;
        let mut buf = [0u8; BYTES];
        self.seek(idx as u64)?;
        let n = self.reader.read(&mut buf)?;

        self.cursor.advance_by(BYTES as u32, mode.sect_size());
        Ok((n, buf))
    }
}

#[derive(derive_more::Debug)]
enum Disc {
    CueBin(CueFormat, DiscReader),
    Raw(DiscReader),
}

impl Disc {
    pub fn seek(&mut self, to: CdromCursor) -> std::io::Result<()> {
        let (padding, reader) = match self {
            Disc::CueBin(cue_format, disc_reader) => {
                (cue_format.index_list[0].second as u32 * 75, disc_reader)
            }
            Disc::Raw(disc_reader) => (0, disc_reader),
        };
        let byte = padding + to.lba_to_bytes();
        reader.seek(byte as u64)
    }

    pub fn readn<const BYTES: usize>(
        &mut self,
        mode: SetMode,
    ) -> std::io::Result<(usize, [u8; BYTES])> {
        match self {
            Disc::CueBin(cue_format, disc_reader) => disc_reader.readn(mode),
            Disc::Raw(disc_reader) => todo!(),
        }
    }
}
