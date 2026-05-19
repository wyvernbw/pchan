mod cdrom_cmds;
mod cdrom_ver;

use crate::{
    Bus, Emu,
    io::{
        CastIOFrom, CastIOInto, UnhandledIO,
        cdrom::cdrom_ver::CDRomVerPtr,
        irq::{self, Interrupts},
    },
    trace_todo,
};
use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use pchan_utils::hex;

#[derive(Default, derive_more::Debug, Clone)]
pub struct CDRomState {
    status:      CDRomStatusReg,
    hint_status: CDRomHIntSts,
    hint_mask:   CDRomHIntMask,
    param_fifo:  heapless::Deque<u8, 16>,
    result_fifo: heapless::Deque<u8, 16>,
    ver:         CDRomVerPtr,
}

/// Current todo:
///
/// - [x] W status reg
/// - [x] W CD Irq flag
/// - [x] W CD Irq on/off
/// - [x] R status reg
/// - [x] W param fifo
/// - [x] W CD cmd reg
/// - [x] R CD Irq flag
/// - [x] R res fifo
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
///
/// log #2:
///
/// ```log
///  WARN pchan_emu::io::cdrom: todo(cdrom): write to param fifo
///  WARN pchan_emu::io::cdrom: todo(cdrom): write to cd command register
/// ````
///
/// log #3:
/// ```log
///  WARN pchan_emu::io::cdrom: todo(cdrom): read from cd irq flag register
/// ```
///
/// log #4:
/// ```log
/// WARN pchan_emu::io::cdrom: todo(cdrom): read from response fifo
/// WARN pchan_emu::io::cdrom::cdrom_cmds: todo(cdrom): unhandled cmd: 0x01
/// ```
pub trait CDRom: Bus + Interrupts {
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

            (0x1f801801, 0) => {
                match self.cdrom_mut().send_cmd(value) {
                    cdrom_cmds::CdromIrqEvent::None => {}
                    cdrom_cmds::CdromIrqEvent::Immediate => {
                        self.trigger_irq(irq::Irq::Irq2CDRom);
                        tracing::info!("trigger cdrom irq!");
                    }
                    cdrom_cmds::CdromIrqEvent::InCycles(_) => todo!(),
                }
                tracing::info!("cdrom = {:#?}", self.cdrom());
                Ok(())
            }
            (0x1f801801, 1) => Ok(()), // unused
            (0x1f801801, 2) => Ok(()), // unused
            (0x1f801801, 3) => {
                trace_todo!(
                    "todo(cdrom): write to cd audio volume for right-cd-out to right-spu-in"
                )
            }

            (0x1f801802, 0) => {
                self.cdrom_mut().param_push(value);
                Ok(())
            }
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
        .inspect(|_| tracing::info!("w(cdrom): {}", hex(address)))
    }
    fn read<T>(&mut self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank();
        match (address, bank) {
            (0x1f801800, _) => Ok(self.cdrom().status.io_from_u32()),
            (0x1f801801, _) => match self.cdrom_mut().result_pop() {
                Some(value) => Ok(value.io_from_u32()),
                // technically this is not correct, see psx spx
                // its probably fine doe
                None => Ok(0.io_from_u32()),
            },

            (0x1f801802, _) => trace_todo!(0u32, "todo(cdrom): read from data fifo"),

            (0x1f801803, 0 | 2) => {
                trace_todo!(0u32, "todo(cdrom): read from cd irq on/off register")
            }
            (0x1f801803, 1 | 3) => Ok(self.cdrom().hint_status.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
        .inspect(|_| tracing::info!("r(cdrom): {}", hex(address)))
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
#[bitfield(u8, debug)]
struct CDRomStatusReg {
    #[bits(0..=1, rw)]
    bank:          u2,
    #[bit(2, rw)]
    adpcm_busy:    bool,
    #[bit(3, rw)]
    param_empty:   bool,
    #[bit(4, rw)]
    param_wready:  bool,
    #[bit(5, rw)]
    result_rready: bool,
    #[bit(6, rw)]
    data_req:      bool,
    #[bit(7, rw)]
    busy_status:   bool,
}

impl Default for CDRomStatusReg {
    fn default() -> Self {
        Self::new_with_raw_value(0x0)
            .with_bank(0.as_())
            .with_adpcm_busy(false)
            .with_param_empty(true)
            .with_param_wready(true)
            .with_result_rready(false)
            .with_data_req(false)
            .with_busy_status(false)
    }
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
    intsts:    Int,
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
    intsts:    Int,
    #[bit(3, rw)]
    buf_empty: bool,
    #[bit(4, rw)]
    buf_wrdy:  bool,
    #[bits(5..=7)]
    _reserved: u3,
}

/// ```plaintext
/// INT0 NoIntr      No interrupt pending
/// INT1 DataReady   New sector (ReadN/ReadS) or report packet (Play) available
/// INT2 Complete    Command finished processing (some commands, after INT3 is fired)
/// INT3 Acknowledge Command received and acknowledged (all commands)
/// INT4 DataEnd     Reached end of disc (or end of track if auto-pause enabled)
/// INT5 DiskError   Command error, read error, license string error or lid opened
/// INT6 -
/// INT7 -
/// ```
#[bitenum(u3, exhaustive = true)]
#[derive(Debug)]
#[expect(clippy::enum_variant_names)]
enum Int {
    Int0NoInt     = 0x0,
    Int1DataReady = 0x1,
    Int2Complete  = 0x2,
    Int3Ack       = 0x3,
    Int4DataEnd   = 0x4,
    Int5DiskErr   = 0x5,
    Int6          = 0x6,
    Int7          = 0x7,
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
    clrint:         Int,
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
        tracing::info!("cdrom: write hardware clear ctl: {hclrctl:#?}");

        {
            let intsts = hintsts.intsts().raw_value();
            let clrint = hclrctl.clrint().raw_value();
            let new_intsts = intsts & !clrint;
            let new_intsts = Int::new_with_raw_value(new_intsts);
            hintsts.set_intsts(new_intsts);
        }

        if hclrctl.clr_buf_empty() {
            hintsts.set_buf_empty(false);
        }
        if hclrctl.clr_buf_wrdy() {
            hintsts.set_buf_wrdy(false);
        }
        if hclrctl.clr_param_fifo() {
            self.param_fifo.clear();
            self.status.set_param_empty(true);
            self.status.set_param_wready(true);
        }
        // TODO: smap, reset decoder
    }
}

impl CDRomState {
    fn param_push(&mut self, param: u8) {
        match self.param_fifo.push_back(param) {
            Ok(()) => {
                self.status.set_param_empty(false);
                if self.param_fifo.is_full() {
                    self.status.set_param_wready(false);
                }
            }
            Err(param) => {
                // overwrite last
                _ = self.param_fifo.pop_back();
                become self.param_push(param);
            }
        }
    }
    fn result_push(&mut self, result: u8) {
        match self.result_fifo.push_back(result) {
            Ok(()) => {
                self.status.set_result_rready(true);
            }
            Err(result) => {
                // overwrite last
                _ = self.result_fifo.pop_back();
                become self.result_push(result);
            }
        }
    }
    fn result_pop(&mut self) -> Option<u8> {
        let res = self.result_fifo.pop_front();
        if self.result_fifo.is_empty() {
            self.status.set_result_rready(false);
        }
        res
    }
}
