#[path = "./cdrom-cmds.rs"]
mod cdrom_cmds;
#[path = "./cdrom-drive.rs"]
mod cdrom_drive;
#[path = "./cdrom-format.rs"]
mod cdrom_format;
#[path = "./cdrom-ver.rs"]
mod cdrom_ver;

use crate::io::cdrom::cdrom_cmds::{CdromResponse, Response};
use crate::io::cdrom::cdrom_drive::{CdromDrive, CommandState, Disc};
use crate::io::cdrom::cdrom_ver::CDRomVerPtr;
use crate::io::evque::{EvCtx, Evque};
use crate::io::irq::{self};
use crate::io::{CastIOFrom, CastIOInto, UnhandledIO};
use crate::{Emu, trace_todo};
use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use pchan_utils::hex;
use slab::Slab;

#[derive(Default, derive_more::Debug, Clone)]
pub struct CDRomState {
    status:      CDRomStatusReg,
    hint_status: CDRomHIntSts,
    hint_mask:   CDRomHIntMask,
    request:     CDRomReqRegister,
    param_fifo:  heapless::Deque<u8, 16>,
    result_fifo: heapless::Deque<u8, 16>,
    data_last:   u8,
    data_fifo:   heapless::Deque<u8, 16>,
    ver:         CDRomVerPtr,

    responses: Slab<Response>,

    drive: CdromDrive,
}

#[derive(Default, derive_more::Debug, Clone)]
enum DriveStatus {
    LidOpen,
    SpinUp,
    DetectBusy,
    #[default]
    NoDisk,
    AudioDisk,
    LicensedMode2,
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
impl Emu {
    #[pchan_macros::pchan_instrument_write]
    pub fn cdrom_write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank();
        let value = value.io_into_u32() as u8;
        match (address, bank) {
            (0x1f801800, _) => {
                let status = CDRomStatusReg::new_with_raw_value(value);
                self.cdrom_mut().status.set_bank(status.bank());
                Ok(())
            }

            (0x1f801801, 0) => {
                for response in self.cdrom_mut().send_cmd(value) {
                    match response {
                        CdromResponse::None => {}
                        CdromResponse::Immediate(response) => {
                            // DONE: dedup this code
                            self.cdrom_send_response(response);

                            self.cdrom.drive.run(&mut CdromScheduler {
                                evque:     &mut self.evque,
                                responses: &mut self.cdrom.responses,
                            });
                        }
                        CdromResponse::InCycles(in_cycles, id) => {
                            self.evque_mut().schedule(
                                Self::handle_ev_cdrom_response,
                                id,
                                in_cycles,
                            );
                        }
                    }
                }
                self.cdrom_mut().param_fifo.clear();
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

            (0x1f801803, 0) => {
                let req = CDRomReqRegister::new_with_raw_value(value);
                self.cdrom.request = req;
                if self.cdrom.request.bfrd() {
                    self.cdrom
                        .drive
                        .request_data(&mut self.cdrom.status, &mut self.cdrom.data_fifo);
                }
                Ok(())
            }
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
        .inspect(|_| tracing::info!("w(cdrom) @ {}:{}", hex(address), bank))
    }

    #[pchan_macros::pchan_instrument_read]
    pub fn cdrom_read<T>(&mut self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        let bank = self.cdrom().bank();
        match (address, bank) {
            (0x1f801800, _) => Ok(self.cdrom().status.io_from_u32()),
            (0x1f801801, _) => match self
                .cdrom_mut()
                .result_pop()
                .inspect(|value| tracing::info!("cdrom: return response {}", hex(*value)))
            {
                Some(value) => Ok(value.io_from_u32()),
                // technically this is not correct, see psx spx
                // its probably fine doe
                None => Ok(0.io_from_u32()),
            },

            (0x1f801802, _) => trace_todo!(0u32, "todo(cdrom): read from data fifo"),

            (0x1f801803, 0 | 2) => Ok(self.cdrom.hint_mask.io_from_u32()),
            (0x1f801803, 1 | 3) => Ok(self.cdrom().hint_status.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
        .inspect(|_| tracing::info!("r(cdrom) @ {}:{}", hex(address), bank))
    }

    fn cdrom_send_response(&mut self, response: Response) {
        self.cdrom_mut().result_push_many(response.data);
        self.cdrom_mut().hint_status.set_intsts(response.int);
        self.cdrom_mut().status.set_busy_status(false);
        let hint_status = self.cdrom().hint_status.raw_value();
        let hint_mask = self.cdrom().hint_mask.raw_value();
        if hint_status & hint_mask != 0 {
            self.irq_trigger(irq::Irq::Irq2CDRom);
        }
    }

    #[tracing::instrument(skip_all)]
    fn handle_ev_cdrom_response(&mut self, ctx: EvCtx) {
        match &self.cdrom.drive.command_state {
            CommandState::Idle => return,
            CommandState::Responding(responses) => {
                if !responses.contains(&ctx.id) {
                    return;
                }
            }
        }

        let response = self.cdrom_mut().responses.remove(ctx.id);
        if response.done {
            self.cdrom.drive.set_command_state(CommandState::Idle);
        }
        self.cdrom_send_response(response);

        tracing::info!("HINT_STAT={}", hex(self.cdrom().hint_status));
        tracing::info!("trigger cdrom irq!");

        self.cdrom.drive.run(&mut CdromScheduler {
            evque:     &mut self.evque,
            responses: &mut self.cdrom.responses,
        });
    }

    pub fn cdrom_read_data<const BYTES: usize>(&mut self) -> [u8; BYTES] {
        let mut buf = [0u8; BYTES];
        for byte in buf.iter_mut() {
            let value = self
                .cdrom
                .data_fifo
                .pop_back()
                .unwrap_or(self.cdrom.data_last);
            self.cdrom.data_last = value;
            *byte = value;
        }
        buf
    }
}

struct CdromScheduler<'a> {
    evque:     &'a mut Evque<Emu>,
    responses: &'a mut Slab<Response>,
}

impl<'a> CdromScheduler<'a> {
    fn schedule(&mut self, in_cycles: u64, res: Response) {
        let res = self.responses.insert(res);
        self.evque
            .schedule(Emu::handle_ev_cdrom_response, res, in_cycles);
    }
}

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
#[bitfield(u8, default = 0x0, debug)]
struct CDRomHIntSts {
    #[bits(0..=2, rw)]
    intsts:    HInt,
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
    intsts:    HInt,
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
#[derive(Debug, PartialEq, Eq)]
pub enum HInt {
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
    clrint:         HInt,
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

/// # 0x1f801803 (write, bank 0): HCHPCTL
///
/// ```plaintext
///   0-4 -    Reserved                                    (should be 0)
///   5   SMEN Sound map (manual XA-ADPCM playback) enable
///   6   BFWR Request sector buffer write                 (1=prepare for writes to WRDATA)
///   7   BFRD Request sector buffer read                  (1=prepare for reads from RDDATA)
/// ```
#[bitfield(u8, debug)]
#[derive(Default)]
struct CDRomReqRegister {
    #[bit(7, rw)]
    bfrd: bool,
}

impl CDRomState {
    fn write_h_clr_ctl(&mut self, hclrctl: CDRomHClrCtl) {
        let hintsts = &mut self.hint_status;

        {
            let intsts = hintsts.intsts().raw_value();
            let clrint = hclrctl.clrint().raw_value();
            let new_intsts = intsts & !clrint;
            let new_intsts = HInt::new_with_raw_value(new_intsts);
            hintsts.set_intsts(new_intsts);

            // if hclrctl.clrint() != HInt::Int0NoInt {
            //     self.result_fifo.clear();
            //     self.status.set_result_rready(false);
            // }
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
                tracing::info!("cdrom.response: pushed {}", hex(result));
                self.status.set_result_rready(true);
            }
            Err(result) => {
                // overwrite last
                _ = self.result_fifo.pop_back();
                become self.result_push(result);
            }
        }
    }
    fn result_push_many(&mut self, results: impl IntoIterator<Item = u8>) {
        for res in results {
            self.result_push(res);
        }
    }
    #[pchan_macros::instrument(skip_all, ret)]
    fn result_pop(&mut self) -> Option<u8> {
        let res = self.result_fifo.pop_front();
        if self.result_fifo.is_empty() {
            self.status.set_result_rready(false);
        }
        res
    }
}
