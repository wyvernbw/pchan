use crate::{
    Bus, Emu,
    io::{CastIOFrom, CastIOInto, UnhandledIO},
};
use arbitrary_int::prelude::*;
use bitbybit::bitfield;

#[derive(Default, derive_more::Debug, Clone)]
pub struct CDRomState {
    status:      CDRomStatusReg,
    hint_status: CDRomHIntSts,
    hint_mask:   CDRomHIntMask,
}

macro_rules! trace_todo {
    ($args: tt) => {{
        tracing::warn!($args);
        Ok(())
    }};
    ($value: expr, $args: tt) => {{
        tracing::warn!($args);
        Ok($value.io_from_u32())
    }};
}

/// Current todo:
///
/// - [x] W status reg
/// - [x] W CD Irq flag
/// - [x] W CD Irq on/off
///
/// log #0:
///
/// ```log
/// WARN pchan_emu::io::cdrom: todo(cdrom): write to status reg
/// WARN pchan_emu::io::cdrom: todo(cdrom): write to request register
/// WARN pchan_emu::io::cdrom: todo(cdrom): write to status reg
/// WARN pchan_emu::io::cdrom: todo(cdrom): write to param fifo
/// ```
///
/// log #1:
///
/// ```log
/// WARN pchan_emu::io::cdrom: todo(cdrom): write to cd irq flag register
/// WARN pchan_emu::io::cdrom: todo(cdrom): write to irq on/off register
/// ```
pub trait CDRom: Bus {
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank();
        let value = value.io_into_u32() as u8;
        match (address, bank) {
            (0x1f801800, _) => {
                let status = CDRomStatusReg::new_with_raw_value(value);
                self.cdrom_mut().status.write(status);
                Ok(())
            }

            (0x1f801801, 0) => trace_todo!("todo(cdrom): write to cd command register"),
            (0x1f801801, 1) => Ok(()), // unused
            (0x1f801801, 2) => Ok(()), // unused
            (0x1f801801, 3) => {
                trace_todo!(
                    "todo(cdrom): write to cd audio volume for right-cd-out to right-spu-in"
                )
            }

            (0x1f801802, 0) => trace_todo!("todo(cdrom): write to param fifo"),
            (0x1f801802, 1) => {
                let hint_mask = CDRomHIntMask::new_with_raw_value(value);
                self.cdrom_mut().hint_mask.write(hint_mask);
                Ok(())
            }
            (0x1f801802, 2) => {
                trace_todo!("todo(cdrom): write to cd audio volume for left-cd-out to left-spu-in")
            }
            (0x1f801802, 3) => {
                trace_todo!("todo(cdrom): write to cd audio volume for right-cd-out to left-spu-in")
            }

            (0x1f801803, 0) => trace_todo!("todo(cdrom): write to request register"),
            (0x1f801803, 1) => {
                let hclrctl = CDRomHClrCtl::new_with_raw_value(value);
                self.cdrom_mut().write_h_clr_ctl(hclrctl);
                Ok(())
            }
            (0x1f801803, 2) => {
                trace_todo!("todo(cdrom): write to cd audio volume for left-cd-out to right-spu-in")
            }
            (0x1f801803, 3) => {
                trace_todo!("todo(cdrom): write to cd audio volume apply")
            }
            _ => Err(UnhandledIO(address)),
        }
    }
    fn read<T>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank();
        match (address, bank) {
            (0x1f801801, _) => trace_todo!(0u32, "todo(cdrom): read from response fifo"),

            (0x1f801802, _) => trace_todo!(0u32, "todo(cdrom): read from data fifo"),

            (0x1f801803, 0 | 2) => {
                trace_todo!(0u32, "todo(cdrom): read from cd irq on/off register")
            }
            (0x1f801803, 1 | 3) => {
                trace_todo!(0u32, "todo(cdrom): read from cd irq flag register")
            }
            _ => Err(UnhandledIO(address)),
        }
    }
}

impl CDRom for Emu {}

/// # `0x1f801800` (write, all banks): ADDRESS
///
/// ```plaintext
/// 0-1 RA       Current register bank (R/W)
/// 2   ADPBUSY  ADPCM busy            (R, 1=playing XA-ADPCM)
/// 3   PRMEMPT  Parameter empty       (R, 1=parameter FIFO empty)
/// 4   PRMWRDY  Parameter write ready (R, 1=parameter FIFO not full)
/// 5   RSLRRDY  Result read ready     (R, 1=result FIFO not empty)
/// 6   DRQSTS   Data request          (R, 1=one or more RDDATA reads or WRDATA writes pending)
/// 7   BUSYSTS  Busy status           (R, 1=HC05 busy acknowledging command)
/// ```
///
/// Writing a value to the low 2 bits of this address changes the bank to said
/// value. Likewise, the low 2 bits of this address can be read to get the current
/// bank.
#[bitfield(u8, debug, default = 0x0)]
struct CDRomStatusReg {
    #[bits(0..=1, rw)]
    bank: u2,
    // TODO: 2..=7
}

impl CDRomStatusReg {
    pub fn write(&mut self, other: Self) {
        self.set_bank(other.bank());
    }
}

impl CDRomState {
    pub fn bank(&self) -> u8 {
        self.status.bank().as_u8()
    }
}

/// # `0x1f801803` (read, banks 1 and 3): HINTSTS
///
/// ```plaintext
///  0-2 INTSTS Interrupt "flags" from HC05
///  3   BFEMPT Sound map XA-ADPCM buffer empty       (1=decoder ran out of sectors to play)
///  4   BFWRDY Sound map XA-ADPCM buffer write ready (1=decoder is ready for next sector)
///  5-7 -      Reserved                              (always 1)
/// ```
#[bitfield(u8, default = 0xe0, debug)]
struct CDRomHIntSts {
    #[bits(0..=2, rw)]
    intsts:    u3,
    #[bit(3, rw)]
    buf_empty: bool,
    #[bit(4, rw)]
    buf_wrdy:  bool,
    #[bits(5..=7)]
    _reserved: u3,
}

#[bitfield(u8, default = 0xe0, debug)]
struct CDRomHIntMask {
    #[bits(0..=2, rw)]
    intsts:    u3,
    #[bit(3, rw)]
    buf_empty: bool,
    #[bit(4, rw)]
    buf_wrdy:  bool,
    #[bits(5..=7)]
    _reserved: u3,
}

impl CDRomHIntMask {
    pub fn write(&mut self, other: Self) {
        self.set_intsts(other.intsts());
        self.set_buf_empty(other.buf_empty());
        self.set_buf_wrdy(other.buf_wrdy());
    }
}

/// # `0x1f801803` (write, bank 1): HCLRCTL
///
/// ```plaintext
///  0-2 CLRINT     Acknowledge HC05 interrupt "flags" (0=no change, 1=clear)
///  3   CLRBFEMPT  Acknowledge BFEMPT                 (0=no change, 1=clear)
///  4   CLRBFWRDY  Acknowledge BFBFWRDY               (0=no change, 1=clear)
///  5   SMADPCLR   Clear sound map XA-ADPCM buffer    (0=no change, 1=clear/stop playback)
///  6   CLRPRM     Clear parameter FIFO               (0=no change, 1=clear)
///  7   CHPRST     Reset decoder chip                 (0=no change, 1=reset)
/// ```
#[bitfield(u8, debug)]
struct CDRomHClrCtl {
    #[bits(0..=2, r)]
    clrint:         u3,
    #[bit(3, r)]
    clr_buf_empty:  bool,
    #[bit(4, r)]
    clr_buf_wrdy:   bool,
    #[bit(5, r)]
    clr_smap:       bool,
    #[bit(6, r)]
    clr_param_fifo: bool,
    #[bit(7, r)]
    reset_decoder:  bool,
}

impl CDRomState {
    fn write_h_clr_ctl(&mut self, hclrctl: CDRomHClrCtl) {
        let hintsts = &mut self.hint_status;
        if hclrctl.clrint().as_u8() != 0 {
            hintsts.set_intsts(0x0.as_());
        }
        if hclrctl.clr_buf_empty() {
            hintsts.set_buf_empty(false);
        }
        if hclrctl.clr_buf_wrdy() {
            hintsts.set_buf_wrdy(false);
        }
        // TODO: rest
    }
}
