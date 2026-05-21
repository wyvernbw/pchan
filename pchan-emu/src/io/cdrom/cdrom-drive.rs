use crate::io::cdrom::cdrom_cmds::SetMode;
use crate::io::cdrom::cdrom_format::{CdromCursor, Mss};
use crate::io::evque::Evque;

use super::cdrom_cmds::SetModeSectSize;

#[derive(Default, derive_more::Debug, Clone)]
pub struct CdromDrive {
    cursor:      CdromCursor,
    sect_size:   u16,
    drive_state: DriveState,
}

#[derive(Default, derive_more::Debug, Clone)]
enum DriveState {
    #[default]
    Idle,
    ReadN,
}

impl CdromDrive {
    pub fn setloc<T>(&mut self, mss: Mss<T>)
    where
        u8: const From<T>,
    {
        self.cursor = CdromCursor::from_mss(mss);
    }

    pub fn setmode(&mut self, setmode: SetMode) {
        if !setmode.ignore_bit() {
            self.sect_size = match setmode.sect_size() {
                SetModeSectSize::DataOnly0x800 => 0x800,
                SetModeSectSize::Whole0x924 => 0x924,
            };
        }
    }

    pub fn readn(&mut self) {
        self.drive_state = DriveState::ReadN;
    }

    pub fn run<T>(&mut self, evque: &mut Evque<T>) {
        todo!()
    }
}
