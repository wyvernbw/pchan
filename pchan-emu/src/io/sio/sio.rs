use arbitrary_int::prelude::*;
use bitbybit::*;
use pchan_utils::hex;

use crate::{
    Bus, Emu,
    io::{CastIOFrom, CastIOInto, UnhandledIO},
    trace_todo,
};

#[derive(Default, derive_more::Debug, Clone)]
pub struct SioState {
    sio0stat: SioStatusReg,
    sio1stat: SioStatusReg,
    sio0ctrl: SioCtrlReg,
}

pub trait Sio: Bus {
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        let value = value.io_into_u32();
        match address {
            // 1/4  JOY_DATA Joypad/Memory Card Data (R/W)
            0x1f801040 => trace_todo!("todo(sio): write to joypad/memcard data"),
            // 1/4  SIO_DATA Serial Port Data (R/W)
            0x1f801050 => trace_todo!("todo(sio): write to sio1 (serial port) data"),

            // status is RO

            // 2    JOY_MODE Joypad/Memory Card Mode (R/W)
            0x1f801048 => trace_todo!("todo(sio): write to joypad/memcard mode"),
            // 2    SIO_MODE Serial Port Mode (R/W)
            0x1f801058 => trace_todo!("todo(sio): write to sio1 (serial port) mode"),

            // 2    JOY_CTRL Joypad/Memory Card Control (R/W)
            0x1f80104a => {
                self.sio_mut()
                    .write_sio0_ctrl(SioCtrlReg::new_with_raw_value(value.io_into_u32()));
                Ok(())
            }
            // 2    SIO_CTRL Serial Port Control (R/W)
            0x1f80105a => trace_todo!("todo(sio): write to sio1 (serial port) ctrl"),

            // 2    JOY_BAUD Joypad/Memory Card Baudrate (R/W)
            0x1f80104e => trace_todo!("todo(sio): write to sio0 bdrate"),
            // 2    SIO_BAUD Serial Port Baudrate (R/W)
            0x1f80105e => trace_todo!("todo(sio): write to sio1 bdrate"),

            // 1F80105Ch 2    SIO_MISC Serial Port Internal Register (R/W)
            0x1f80105c => trace_todo!("todo(sio): write to sio1 (serial port) internal"),
            _ => Err(UnhandledIO(address)),
        }
        .inspect(|_| tracing::info!("w(sio): {}", hex(address)))
    }
    fn read<T>(&mut self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            // 1F801040h 1/4  JOY_DATA Joypad/Memory Card Data (R/W)
            0x1f801040 => trace_todo!(0x0, "todo(sio): read from joypad/memcard data"),
            0x1f801050 => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) data"),

            // 1F801044h 4    JOY_STAT Joypad/Memory Card Status (R)
            0x1f801044 => Ok(self.sio().sio0stat.io_from_u32()),
            0x1f801054 => Ok(self.sio().sio1stat.io_from_u32()),

            // 1F801048h 2    JOY_MODE Joypad/Memory Card Mode (R/W)
            0x1f801048 => trace_todo!(0x0, "todo(sio): read from joypad/memcard mode"),
            0x1f801058 => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) mode"),

            // 1F80104Ah 2    JOY_CTRL Joypad/Memory Card Control (R/W)
            0x1f80104a => trace_todo!(0x0, "todo(sio): read from joypad/memcard ctrl"),
            0x1f80105a => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) ctrl"),

            // 1F80104Eh 2    JOY_BAUD Joypad/Memory Card Baudrate (R/W)
            0x1f80104e => trace_todo!(0x0, "todo(sio): read from joypad/memcard bdrate"),
            0x1f80105e => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) bdrate"),

            // 1F80105Ch 2    SIO_MISC Serial Port Internal Register (R/W)
            0x1f80105c => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) internal"),
            _ => Err(UnhandledIO(address)),
        }
        .inspect(|_| tracing::info!("r(sio): {}", hex(address)))
    }
}

impl Sio for Emu {}

const fn sio_idx(addr: u32, base: u32, stride: u32) -> Option<usize> {
    let addr = addr - base;
    if (addr).is_multiple_of(stride) {
        Some((addr / stride) as usize)
    } else {
        None
    }
}

/// 1F801044h+N*10h - SIO#_STAT (R)
///
/// ```plaintext
///   0     TX FIFO Not Full       (1=Ready for new byte)  (depends on CTS) (TX requires CTS)
///   1     RX FIFO Not Empty      (0=Empty, 1=Data available)
///   2     TX Idle                (1=Idle/Finished)       (depends on TXEN and on CTS)
///   3     RX Parity Error        (0=No, 1=Error; Wrong Parity, when enabled) (sticky)
///   4     SIO1 RX FIFO Overrun   (0=No, 1=Error; received more than 8 bytes) (sticky)
///   5     SIO1 RX Bad Stop Bit   (0=No, 1=Error; Bad Stop Bit) (when RXEN)   (sticky)
///   6     SIO1 RX Input Level    (0=Normal, 1=Inverted) ;only AFTER receiving Stop Bit
///   7     DSR Input Level        (0=Off, 1=On) (remote DTR) ;DSR not required to be on
///   8     SIO1 CTS Input Level   (0=Off, 1=On) (remote RTS) ;CTS required for TX
///   9     Interrupt Request      (0=None, 1=IRQ) (See SIO_CTRL.Bit4,10-12)   (sticky)
///   10    Unknown                (always zero)
///   11-31 Baudrate Timer         (15-21 bit timer, decrementing at 33MHz)
/// ```
///
/// Bit 0 gets set after sending the start bit, bit 2 is set after sending all bits including the stop bit if any.
/// On SIO0, DSR is wired to the /ACK pin on the controller and memory card ports; bit 7 is thus set when /ACK is low (asserted) and cleared when it is high. Bits 4-6 and 8 are always zero.
/// The number of bits actually used by the baud rate timer is probably affected by the reload factor set in SIO_MODE.
#[bitfield(u32, debug, default = 0x0)]
struct SioStatusReg {
    #[bit(0, rw)]
    tx_not_full:  bool,
    #[bit(1, rw)]
    rx_not_empty: bool,
    #[bit(2, rw)]
    tx_idle:      bool,
    #[bit(3, rw)]
    rx_par_err:   bool,

    // sio 1 only
    #[bit(4, rw)]
    sio1_rx_overrun:  bool,
    #[bit(5, rw)]
    sio1_rx_bad_stop: bool,
    #[bit(6, rw)]
    sio1_rx_in_lvl:   bool,
    #[bit(8, rw)]
    sio1_cts_in_lvl:  bool,

    #[bit(7, rw)]
    dsr_in_lvl: bool,
    #[bit(9, rw)]
    irq:        bool,

    #[bits(11..=31, rw)]
    bd_timer: u21,
}

/// # 1F80104Ah+N*10h - SIO#_CTRL (R/W)
///
/// ```paintext
///   0     TX Enable (TXEN)      (0=Disable, 1=Enable)
///   1     DTR Output Level      (0=Off, 1=On)
///   2     RX Enable (RXEN)      (SIO1: 0=Disable, 1=Enable)  ;Disable also clears RXFIFO
///                               (SIO0: 0=only receive when /CS low, 1=force receiving single byte)
///   3     SIO1 TX Output Level  (0=Normal, 1=Inverted, during Inactivity & Stop bits)
///   4     Acknowledge           (0=No change, 1=Reset SIO_STAT.Bits 3,4,5,9)      (W)
///   5     SIO1 RTS Output Level (0=Off, 1=On)
///   6     Reset                 (0=No change, 1=Reset most registers to zero) (W)
///   7     SIO1 unknown?         (read/write-able when FACTOR non-zero) (otherwise always zero)
///   8-9   RX Interrupt Mode     (0..3 = IRQ when RX FIFO contains 1,2,4,8 bytes)
///   10    TX Interrupt Enable   (0=Disable, 1=Enable) ;when SIO_STAT.0-or-2 ;Ready
///   11    RX Interrupt Enable   (0=Disable, 1=Enable) ;when N bytes in RX FIFO
///   12    DSR Interrupt Enable  (0=Disable, 1=Enable) ;when SIO_STAT.7  ;DSR high or /ACK low
///   13    SIO0 port select      (0=port 1, 1=port 2) (/CS pulled low when bit 1 set)
///   14-15 Not used              (always zero)
/// ```
///
/// On SIO0, DTR is wired to the /CS pin on the controller and memory card ports; bit 1 will pull (assert) /CS low when set. Bit 13 is used to select which port's /CS shall be asserted (all other signals are wired in parallel).
/// Bit 2 behaves differently on SIO0: when not set, incoming data will be ignored unless bit 1 is also set. When set, data will be received regardless of whether /CS is asserted, however bit 2 will be automatically cleared after a byte is received.
/// Note that some emulators do not implement all SIO0 interrupts, as the kernel's controller driver only ever uses the DSR (/ACK) interrupt.
#[bitfield(u32, debug, default = 0x0)]
struct SioCtrlReg {
    #[bit(0, rw)]
    tx_on:       bool,
    #[bit(1)]
    dtr_out_lvl: bool,
    #[bit(2, rw)]
    rx_on:       bool,

    // sio1 only
    #[bit(3, rw)]
    sio1_tx_out_lvl:  bool,
    #[bit(5, rw)]
    sio1_rts_out_lvl: bool,

    #[bit(4, rw)]
    ack:   bool,
    #[bit(6)]
    reset: bool,

    #[bits(8..=9, rw)]
    rx_irq_mode: RxIrqMode,

    #[bit(10, rw)]
    tx_irq_on:  bool,
    #[bit(11, rw)]
    rx_irq_on:  bool,
    #[bit(12, rw)]
    dsr_irq_on: bool,

    // sio0 only
    #[bit(13)]
    sio0_port: Sio0Port,
}

#[bitenum(u2, exhaustive = true)]
#[expect(clippy::enum_variant_names)]
#[derive(Debug)]
enum RxIrqMode {
    M1Byte = 0x0,
    M2Byte = 0x1,
    M4Byte = 0x2,
    M8Byte = 0x3,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum Sio0Port {
    Port1 = 0x0,
    Port2 = 0x1,
}

impl SioState {
    fn write_sio0_ctrl(&mut self, ctrl: SioCtrlReg) {
        if ctrl.ack() {
            self.sio0stat.set_rx_par_err(false);
            self.sio0stat.set_irq(false);
        }
    }
}
