mod joypad;

use arbitrary_int::prelude::*;
use bitbybit::*;
use derive_more as d;
use heapless::{Deque, binary_heap::Min};
use pchan_utils::hex;

use crate::{
    Bus, Emu,
    io::{
        CastIOFrom, CastIOInto, UnhandledIO,
        irq::{Interrupts, Irq},
        sio::joypad::Joypad,
    },
    trace_todo,
};

use super::irq;

const HI_Z: u32 = u32::MAX;

#[derive(Default, derive_more::Debug, Clone)]
pub struct SioState {
    sio0stat: SioStatusReg,
    sio1stat: SioStatusReg,
    sio0ctrl: SioCtrlReg,
    sio1ctrl: SioCtrlReg,
    sio0mode: SioModeReg,
    sio1mode: SioModeReg,

    sio0bdrate_reload: u16,

    sio0_rx: Sio0Rx,
    sio0_tx: Sio0Tx,

    event_queue: heapless::BinaryHeap<ScheduledSioEvent, Min, 8>,

    sio0devices: Sio0Ports,
    irq_latch:   bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SioEvent {
    Sio0ProcTx,
    Sio0Irq,
    Sio0Ack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScheduledSioEvent {
    clock: u64,
    event: SioEvent,
}

impl PartialOrd for ScheduledSioEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledSioEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.clock.cmp(&other.clock)
    }
}

#[derive(derive_more::Debug, Clone, Copy)]
pub enum PeripheralKind {
    Joypad,
}

#[derive(derive_more::Debug, Clone)]
struct Sio0Ports {
    selected: Option<PeripheralKind>,
    ports:    [PortState; 2],
}

impl Default for Sio0Ports {
    fn default() -> Self {
        Self {
            selected: Default::default(),
            ports:    [
                // connect joypad on port 1 by default
                PortState {
                    joypad:  Joypad::default().plug_in(),
                    memcard: (),
                },
                PortState::default(),
            ],
        }
    }
}

impl Sio0Ports {
    fn is_connected(&self, on_port: Sio0Port) -> bool {
        match self.selected {
            Some(PeripheralKind::Joypad) => self.port(on_port).joypad.connected,
            None => false,
        }
    }
}

#[derive(Default, derive_more::Debug, Clone)]
struct PortState {
    joypad:  Joypad,
    // TODO
    memcard: (),
}

pub enum TxWriteResult {
    Ok,
    TransferFinished,
}

pub trait Peripheral {
    fn on_tx_write(&mut self, byte: u8, rx: &mut Sio0Rx) -> TxWriteResult;
}

#[derive(d::Deref, d::DerefMut, Debug, Default, Clone)]
pub struct Sio0Rx(Deque<u8, 4>);

pub trait Sio: Bus + Interrupts {
    #[pchan_macros::instrument(skip_all, fields(pc = %hex(self.cpu().pc)))]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        let value = value.io_into_u32();
        match address {
            // 1/4  JOY_DATA Joypad/Memory Card Data (R/W)
            0x1f801040 => {
                let value = value as u8;
                self.sio_mut().sio0_tx_send(value);
                Ok(())
            }
            // 1/4  SIO_DATA Serial Port Data (R/W)
            0x1f801050 => trace_todo!("todo(sio): write to sio1 (serial port) data"),

            // status is RO
            0x1f801044 => Ok(()),
            0x1f801054 => Ok(()),

            // 2    JOY_MODE Joypad/Memory Card Mode (R/W)
            0x1f801048 => {
                self.sio_mut().sio0mode = SioModeReg::new_with_raw_value(value as u16);
                Ok(())
            }
            // 2    SIO_MODE Serial Port Mode (R/W)
            0x1f801058 => trace_todo!("todo(sio): write to sio1 (serial port) mode"),

            // 2    JOY_CTRL Joypad/Memory Card Control (R/W)
            0x1f80104a => {
                self.sio_mut()
                    .write_sio0_ctrl(SioCtrlReg::new_with_raw_value(value.io_into_u32()));
                if let Some((event, in_cycles)) = self.sio_mut().sio0_run_transfer() {
                    self.schedule(event, in_cycles);
                }
                Ok(())
            }
            // 2    SIO_CTRL Serial Port Control (R/W)
            0x1f80105a => trace_todo!("todo(sio): write to sio1 (serial port) ctrl"),

            // 2    JOY_BAUD Joypad/Memory Card Baudrate (R/W)
            0x1f80104e => {
                self.sio_mut().sio0bdrate_reload = value.io_into_u32() as u16;
                Ok(())
            }
            // 2    SIO_BAUD Serial Port Baudrate (R/W)
            0x1f80105e => trace_todo!("todo(sio): write to sio1 bdrate"),

            // 1F80105Ch 2    SIO_MISC Serial Port Internal Register (R/W)
            0x1f80105c => trace_todo!("todo(sio): write to sio1 (serial port) internal"),
            _ => Err(UnhandledIO(address)),
        }
        // .inspect(|_| tracing::info!("w(sio): {}", hex(address)))
    }
    #[pchan_macros::instrument(skip_all, fields(pc = %hex(self.cpu().pc)))]
    fn read<T: Copy>(&mut self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            // 1F801040h 1/4  JOY_DATA Joypad/Memory Card Data (R/W)
            0x1f801040 => {
                let og_len = self.sio().sio0_rx.len();
                let og_rx_not_empty = self.sio().sio0stat.rx_not_empty();
                // the sio hardware is a piece of shit so reading 2 bytes
                // removes only one from rx, but reading 4 removes all 4!
                let cnt = size_of::<T>();
                let res: u32 = match cnt {
                    1 => self
                        .sio_mut()
                        .sio0_rx
                        .pop_front()
                        .map(|value| value as u32)
                        .unwrap_or(HI_Z),
                    2 => {
                        let rx = &mut self.sio_mut().sio0_rx;
                        let buf = [
                            rx.pop_front().unwrap_or(0xff),
                            *rx.front().unwrap_or(&0xff),
                            0xff,
                            0xff,
                        ];
                        u32::from_ne_bytes(buf)
                    }
                    4 => {
                        let mut buf = [0xffu8; 4];
                        for x in buf.iter_mut() {
                            let Some(val) = self.sio_mut().sio0_rx.pop_front() else {
                                break;
                            };
                            *x = val;
                        }
                        u32::from_ne_bytes(buf)
                    }
                    _ => 0xff,
                };
                tracing::info!(
                    "sio0: recv {} (len={}, sio_stat.1={})",
                    hex(res),
                    og_len,
                    og_rx_not_empty
                );
                let empty = self.sio().sio0_rx.is_empty();
                self.sio_mut().sio0stat.set_rx_not_empty(!empty);
                Ok(res.io_from_u32::<T>())
            }
            // 1F801044h 4    JOY_STAT Joypad/Memory Card Status (R)
            0x1f801044 => {
                // self.sio_mut().sio0stat.set_dsr_in_lvl(false);
                let empty = self.sio().sio0_rx.is_empty();
                self.sio_mut().sio0stat.with_rx_not_empty(!empty);
                Ok(self.sio().sio0stat.io_from_u32())
            }
            _ => Sio::read_pure(self, address),
        }
        // .inspect(|value| {
        //     tracing::info!(
        //         "r(sio): [{}]={} ({} byte - 0 extended)",
        //         hex(address),
        //         hex((*value).io_into_u32()),
        //         size_of::<T>(),
        //     )
        // })
    }
    fn read_pure<T: Copy>(&self, address: u32) -> Result<T, UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801040 => Ok(0xcafebabeu32.io_from_u32::<T>()),
            0x1f801050 => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) data"),

            0x1f801054 => Ok(self.sio().sio1stat.io_from_u32()),

            // 1F801048h 2    JOY_MODE Joypad/Memory Card Mode (R/W)
            0x1f801048 => trace_todo!(0x0, "todo(sio): read from joypad/memcard mode"),
            0x1f801058 => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) mode"),

            // 1F80104Ah 2    JOY_CTRL Joypad/Memory Card Control (R/W)
            0x1f80104a => Ok(self.sio().read_sio0_ctrl().io_from_u32()),
            0x1f80105a => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) ctrl"),

            // 1F80104Eh 2    JOY_BAUD Joypad/Memory Card Baudrate (R/W)
            0x1f80104e => trace_todo!(0x0, "todo(sio): read from joypad/memcard bdrate"),
            0x1f80105e => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) bdrate"),

            // 1F80105Ch 2    SIO_MISC Serial Port Internal Register (R/W)
            0x1f80105c => trace_todo!(0x0, "todo(sio): read from sio1 (serial port) internal"),
            _ => Err(UnhandledIO(address)),
        }
    }

    fn schedule(&mut self, event: SioEvent, in_cycles: u64) {
        let event = ScheduledSioEvent {
            clock: self.cpu().cycles.wrapping_add(in_cycles),
            event,
        };
        _ = self.sio_mut().event_queue.push(event);
    }

    fn run_sio_io(&mut self, d_clock: u64) {
        let sio = self.sio_mut();
        {
            let bd = sio.sio0stat.bd_timer().as_u32();
            match bd.checked_sub(d_clock as u32) {
                Some(bd) => {
                    sio.sio0stat.set_bd_timer(bd.as_());
                }
                None => {
                    sio.sio0stat.set_bd_timer(sio.sio_cycles().as_());
                }
            }
        }
        if sio.sio0stat.irq() && !sio.irq_latch {
            sio.irq_latch = true;
            self.schedule(SioEvent::Sio0Irq, 100);
        }

        if let Some((event, cycles)) = self.sio_mut().sio0_run_transfer() {
            self.schedule(event, cycles);
        }

        while let Some(event) = self.sio_mut().event_queue.pop() {
            if event.clock > self.cpu().cycles {
                _ = self.sio_mut().event_queue.push(event);
                break;
            }
            match event.event {
                SioEvent::Sio0ProcTx => {
                    self.sio0_tx_proc();
                }
                SioEvent::Sio0Irq => {
                    self.sio_mut().irq_latch = false;
                    let cycles = self.sio().sio_cycles();
                    self.trigger_irq(Irq::Irq7JoypadAndMemcard);
                    self.schedule(SioEvent::Sio0Ack, cycles as _);
                }
                SioEvent::Sio0Ack => {
                    self.sio_mut().sio0stat.set_dsr_in_lvl(false);
                    tracing::info!("pulse /ack");
                }
            };
        }
    }

    fn sio0_tx_proc(&mut self) {
        let Some(value) = self.sio_mut().sio0_tx_pop() else {
            return;
        };

        debug_assert!(self.sio().sio0stat.tx_not_full(), "tx must be mid transfer");

        let tx_write_result: Option<TxWriteResult> =
            match (self.sio().sio0_selected_device(), value) {
                (None, 0x01) => {
                    self.sio0_select(PeripheralKind::Joypad);
                    let ctrl = self.sio().sio0ctrl;
                    if ctrl.rx_on() | ctrl.dtr_out_lvl() {
                        self.sio_mut().sio0_rx.push(HI_Z as u8);
                    }
                    Some(TxWriteResult::Ok)
                }
                (None, _) => return,
                (Some(device), byte) => {
                    let port = self.sio().sio0ctrl.sio0_port();
                    if self.sio().sio0devices.is_connected(port) {
                        let port = self.sio_mut().view_serial_port(port);
                        match device {
                            PeripheralKind::Joypad => {
                                Some(port.port.joypad.on_tx_write(byte, port.rx))
                            }
                        }
                    } else {
                        None
                    }
                }
            };
        let rx_empty = self.sio().sio0_rx.is_empty();
        self.sio_mut().sio0stat.set_rx_not_empty(!rx_empty);
        if let Some(tx_write_result) = tx_write_result {
            match tx_write_result {
                TxWriteResult::Ok => {
                    self.sio_mut().sio0stat.set_tx_idle(true);
                    let port = self.sio().sio0ctrl.sio0_port();
                    if self.sio().sio0devices.is_connected(port) {
                        if self.sio().sio0ctrl.dsr_irq_on() && self.sio().sio0ctrl.tx_on() {
                            if !self.sio().sio0stat.dsr_in_lvl() {
                                self.sio_mut().sio0stat.set_irq(true);
                            }
                            self.sio_mut().sio0stat.set_dsr_in_lvl(true);
                        }
                    }
                }
                TxWriteResult::TransferFinished => {
                    self.sio0_deselect();
                }
            }
        }

        tracing::info!("SIO0_STAT.1" = self.sio().sio0stat.rx_not_empty())
    }

    fn sio0_select(&mut self, device: PeripheralKind) {
        self.sio_mut().sio0devices.selected = Some(device);
    }

    fn sio0_deselect(&mut self) {
        self.sio_mut().sio0devices.selected = None;
    }

    fn sio0_rx_send(&mut self, value: u8) {
        self.sio_mut().sio0_rx.push(value);
        let rx_len = 1usize << (self.sio().sio0ctrl.rx_irq_mode() as usize);
        if self.sio().sio0_rx.len() == rx_len && self.sio().sio0ctrl.rx_irq_on() {
            self.trigger_irq(irq::Irq::Irq7JoypadAndMemcard);
        }
    }
}

struct SerialPortViewMut<'a> {
    port: &'a mut PortState,
    rx:   &'a mut Sio0Rx,
}

impl SioState {
    fn view_serial_port(&mut self, port: Sio0Port) -> SerialPortViewMut<'_> {
        SerialPortViewMut {
            port: &mut self.sio0devices.ports[port as usize],
            rx:   &mut self.sio0_rx,
        }
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
/// Bit 0 gets set after sending the start bit, bit 2 is set after sending all
/// bits including the stop bit if any. On SIO0, DSR is wired to the /ACK pin
/// on the controller and memory card ports; bit 7 is thus set when /ACK is low
/// (asserted) and cleared when it is high. Bits 4-6 and 8 are always zero. The
/// number of bits actually used by the baud rate timer is probably affected by
/// the reload factor set in SIO_MODE.
#[bitfield(u32, debug, default = 0b001)]
struct SioStatusReg {
    /// SIO_STAT.0
    #[bit(0, rw)]
    tx_not_full:  bool,
    /// SIO_STAT.1
    #[bit(1, rw)]
    rx_not_empty: bool,
    /// SIO_STAT.2
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

    /// SIO_STAT.7
    #[bit(7, rw)]
    dsr_in_lvl: bool,
    /// SIO_STAT.9
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
/// On SIO0, DTR is wired to the /CS pin on the controller and memory card
/// ports; bit 1 will pull (assert) /CS low when set. Bit 13 is used to
/// select which port's /CS shall be asserted (all other signals are wired in
/// parallel). Bit 2 behaves differently on SIO0: when not set, incoming data
/// will be ignored unless bit 1 is also set. When set, data will be received
/// regardless of whether /CS is asserted, however bit 2 will be automatically
/// cleared after a byte is received. Note that some emulators do not implement
/// all SIO0 interrupts, as the kernel's controller driver only ever uses the
/// DSR (/ACK) interrupt.
#[bitfield(u32, debug, default = 0x0)]
struct SioCtrlReg {
    #[bit(0, rw)]
    tx_on:       bool,
    #[bit(1, rw)]
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
    #[bit(6, rw)]
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
    #[bit(13, rw)]
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
#[derive(Debug, PartialEq, Eq)]
enum Sio0Port {
    Port1 = 0x0,
    Port2 = 0x1,
}

/// # 1F801048h+N*10h - SIO#_MODE (R/W) (eg. 004Eh --> 8N1 with Factor=MUL16)
///
/// ```plaintext
///   0-1   Baudrate Reload Factor     (1=MUL1, 2=MUL16, 3=MUL64) (or 0=MUL1 on SIO0, STOP on SIO1)
///   2-3   Character Length           (0=5 bits, 1=6 bits, 2=7 bits, 3=8 bits)
///   4     Parity Enable              (0=No, 1=Enable)
///   5     Parity Type                (0=Even, 1=Odd) (seems to be vice-versa...?)
///   6-7   SIO1 stop bit length       (0=Reserved/1bit, 1=1bit, 2=1.5bits, 3=2bits)
///   8     SIO0 clock polarity (CPOL) (0=High when idle, 1=Low when idle)
///   9-15  Not used (always zero)
/// ```
///
/// Bits 6-7 on SIO0 and bit 8 on SIO1 are always zero. On SIO0 the character
/// length shall be set to 8, the clock polarity should be set to high-when-idle
/// and parity should be disabled, as all controllers and memory cards expect these
/// settings.
#[bitfield(u16, debug, default = 0x0)]
struct SioModeReg {
    #[bits(0..=1, rw)]
    bdrate_reload_factor: BdrateReloadFactor,

    #[bits(2..=3, rw)]
    char_len:  CharLen,
    #[bit(4, rw)]
    parity_on: bool,
    #[bit(5, rw)]
    parity_ty: ParityType,

    #[bits(6..=7, rw)]
    sio1_stop_bit_len:   Sio1StopBitLen,
    #[bit(8, rw)]
    sio0_clock_polarity: Sio0ClockPolarity,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug)]
enum BdrateReloadFactor {
    Mul1OrStop = 0x0,
    Mul1       = 0x1,
    Mul16      = 0x2,
    Mul64      = 0x3,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug)]
#[expect(clippy::enum_variant_names)]
enum CharLen {
    L5Bits = 0x0,
    L6Bits = 0x1,
    L7Bits = 0x2,
    L8Bits = 0x3,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum ParityType {
    Even = 0x0,
    Odd  = 0x1,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug)]
enum Sio1StopBitLen {
    LenNA     = 0x0,
    Len1Bit   = 0x1,
    Len1_5Bit = 0x2,
    Len2Bit   = 0x3,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum Sio0ClockPolarity {
    High = 0x0,
    Low  = 0x1,
}

#[derive(Debug, Clone, Copy, Default)]
enum Sio0Tx {
    #[default]
    Idle,
    Queued(u8),
    Transferring(u8, Option<u8>),
}

impl SioState {
    fn sio0_tx_send(&mut self, value: u8) {
        self.sio0stat.set_tx_idle(false);
        self.sio0stat.set_tx_not_full(false);
        tracing::info!("sio: send {}", hex(value));
        match self.sio0_tx {
            Sio0Tx::Idle => {
                self.sio0_tx = Sio0Tx::Queued(value);
            }
            Sio0Tx::Queued(_) => {
                self.sio0_tx = Sio0Tx::Queued(value);
            }
            Sio0Tx::Transferring(old_value, _) => {
                self.sio0stat.set_tx_not_full(true);
                self.sio0_tx = Sio0Tx::Transferring(old_value, Some(value));
            }
        }
    }

    fn sio0_run_transfer(&mut self) -> Option<(SioEvent, u64)> {
        if self.sio0ctrl.tx_on() {
            match self.sio0_tx {
                Sio0Tx::Idle => {}
                Sio0Tx::Queued(value) => {
                    self.sio0_tx = Sio0Tx::Transferring(value, None);
                    self.sio0stat.set_tx_not_full(true);
                    return Some((SioEvent::Sio0ProcTx, 100));
                }
                Sio0Tx::Transferring(_, _) => {}
            }
        } else if let Sio0Tx::Transferring(_, Some(next)) = self.sio0_tx {
            self.sio0_tx = Sio0Tx::Queued(next)
        }

        None
    }

    fn sio0_tx_pop(&mut self) -> Option<u8> {
        match self.sio0_tx {
            Sio0Tx::Idle => None,
            Sio0Tx::Queued(_) => None,
            Sio0Tx::Transferring(value, Some(next)) => {
                self.sio0_tx = Sio0Tx::Transferring(next, None);
                self.sio0stat.set_tx_not_full(true);
                Some(value)
            }
            Sio0Tx::Transferring(value, None) => {
                self.sio0_tx = Sio0Tx::Idle;
                self.sio0stat.set_tx_not_full(true);
                Some(value)
            }
        }
    }

    fn write_sio0_ctrl(&mut self, ctrl: SioCtrlReg) {
        // tracing::info!("sio0: write ctrl {ctrl:#?}");
        let old_port = self.sio0ctrl.sio0_port();
        self.sio0ctrl = ctrl;
        let port = self.sio0ctrl.sio0_port();
        if old_port != port || !self.sio0ctrl.dtr_out_lvl() {
            self.sio0devices.selected = None;
            self.sio0_tx = Sio0Tx::Idle;
        }

        tracing::info!(ack=ctrl.ack(), port = ?self.sio0ctrl.sio0_port(), "/CS"=self.sio0ctrl.dtr_out_lvl(), rxen = ?(self.sio0ctrl.rx_on() as usize));
        if ctrl.ack() {
            self.sio0stat.set_rx_par_err(false);
            self.sio0stat.set_irq(false);
        }

        if ctrl.reset() {
            self.sio0mode = SioModeReg::default();
            self.sio0stat = SioStatusReg::default();
            self.sio0ctrl = SioCtrlReg::default();
            self.sio0_tx = Sio0Tx::Idle;
            self.sio0_rx.clear();
        }

        self.sio0ctrl.set_ack(false);
        self.sio0ctrl.set_reset(false);
    }

    fn read_sio0_ctrl(&self) -> SioCtrlReg {
        tracing::info!(port = ?self.sio0ctrl.sio0_port(), "/CS"=self.sio0ctrl.dtr_out_lvl(), rxen = ?(self.sio0ctrl.rx_on() as usize));
        self.sio0ctrl
    }

    fn sio0_selected_device(&self) -> Option<PeripheralKind> {
        let port = self.sio0ctrl.sio0_port();
        let port = self.sio0devices.port(port);
        let selected = self.sio0devices.selected?;
        match selected {
            PeripheralKind::Joypad => port.joypad.connected.then_some(selected),
        }
    }

    fn sio_cycles(&self) -> u32 {
        let bd = self.sio0bdrate_reload as u32;
        let factor = match self.sio0mode.bdrate_reload_factor() {
            BdrateReloadFactor::Mul1OrStop => 1,
            BdrateReloadFactor::Mul1 => 1,
            BdrateReloadFactor::Mul16 => 16,
            BdrateReloadFactor::Mul64 => 64,
        };
        let bd = bd * factor;

        bd / 2
    }
}

impl Sio0Ports {
    fn port(&self, port: Sio0Port) -> &PortState {
        &self.ports[port as usize]
    }
}

impl Sio0Rx {
    /// overwrites the last item if full
    fn push(&mut self, value: u8) {
        if self.is_full() {
            self.pop_back();
        }
        let res = self.push_back(value);
        debug_assert!(res.is_ok(), "cannot be full")
    }
}
