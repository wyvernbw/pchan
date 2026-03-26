use crate::{
    Bus, Emu,
    gpu::Gpu,
    io::{CastIOFrom, CastIOInto, IO, IOResult, UnhandledIO},
    memory::fastmem::Fastmem,
};
use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use heapless::binary_heap::Min;
use pchan_macros::{pchan_instrument_read, pchan_instrument_write};

#[derive(Debug, Clone)]
pub struct DmaState {
    dpcr: Dpcr,
    dicr: Dicr,
    // TODO: dma channels
    dma2: DmaChannel,
    dma6: DmaChannel,

    queue: DmaQueue,
}

impl Default for DmaState {
    fn default() -> Self {
        Self {
            dpcr:  Dpcr::new_with_raw_value(0x07654321),
            dicr:  Dicr::default(),
            dma2:  DmaChannel::default(),
            dma6:  DmaChannel::default(),
            queue: DmaQueue::default(),
        }
    }
}

///
/// # DMA Register Summary
///
///   1F80108xh DMA0 channel 0  MDECin  (RAM to MDEC)
///   1F80109xh DMA1 channel 1  MDECout (MDEC to RAM)
///   1F8010Axh DMA2 channel 2  GPU (lists + image data)
///   1F8010Bxh DMA3 channel 3  CDROM   (CDROM to RAM)
///   1F8010Cxh DMA4 channel 4  SPU
///   1F8010Dxh DMA5 channel 5  PIO (Expansion Port)
///   1F8010Exh DMA6 channel 6  OTC (reverse clear OT) (GPU related)
///   1F8010F0h DPCR - DMA Control register
///   1F8010F4h DICR - DMA Interrupt register
///
/// These ports control DMA at the CPU-side. In most cases, you'll additionally
/// need to initialize an address (and transfer direction, transfer enabled, etc.)
/// at the remote-side (eg. at the GPU-side for DMA2).
pub trait Dma: Bus + IO + Fastmem {
    fn dma(&self) -> &DmaState;
    fn dma_mut(&mut self) -> &mut DmaState;
    #[pchan_instrument_read("dma:r")]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801080..=0x1f80108f => todo!("read at dma0 (MDECin)"),
            0x1f801090..=0x1f80109f => todo!("read at dma1 (MDECout)"),
            // 0x1f8010a0..=0x1f8010af => todo!("read at dma2 (gpu)"),

            // dma 2
            0x1f8010a0 => todo!("read at dma2madr (gpu madr)"),
            0x1f8010a4 => todo!("read at dma2bcr (gpu bcr)"),
            0x1f8010a8 => {
                let chcr = self.dma().dma2.chcr;
                tracing::trace!("read at dma2chcr (gpu chcr): {:#?}", chcr);
                Ok(chcr.io_from_u32())
            }

            0x1f8010b0..=0x1f8010bf => todo!("read at dma3 (cdrom)"),
            0x1f8010c0..=0x1f8010cf => todo!("read at dma4 (spu)"),
            0x1f8010d0..=0x1f8010df => todo!("read at dma5 (pio)"),

            // dma 6
            0x1f8010e0 => todo!("read at dma6madr (otc madr)"),
            0x1f8010e4 => todo!("read at dma6bcr (otc bcr)"),
            0x1f8010e8 => {
                let chcr = self.dma().dma6.chcr;
                tracing::trace!("read at dma6chcr (otc chcr): {:#?}", chcr);
                Ok(chcr.io_from_u32())
            }

            0x1f8010f0 => Ok(self.dma().dpcr.io_from_u32()),
            0x1f8010f4 => Ok(self.dma().dicr.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[pchan_instrument_write("dma:w")]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801080..=0x1f80108f => todo!("write at dma0 (MDECin)"),
            0x1f801090..=0x1f80109f => todo!("write at dma1 (MDECout)"),

            // dma 2
            0x1f8010a0 => {
                self.dma_mut().dma2.io_set_madr(value);
                tracing::trace!("write at dma2madr (gpu madr): {:#?}", self.dma().dma2.madr);
                Ok(())
            }
            0x1f8010a4 => {
                self.dma_mut().dma2.io_set_bcr(value);
                tracing::trace!("write at dma2bcr (gpu bcr): {:#?}", self.dma().dma2.madr);
                Ok(())
            }
            0x1f8010a8 => {
                let chcr = DmaChcr::new_with_raw_value(value.io_into_u32());
                self.dma_mut().dma2.chcr = chcr;
                tracing::trace!("write at dma2chcr (gpu chcr): {:#?}", chcr);
                match self.dma().dma2.chcr.transfer() {
                    Transfer::StoppedCompleted => {}
                    Transfer::StartBusy => {
                        let gp0cmds = GP0Cmds {
                            init_chan: self.dma().dma2,
                        };
                        let cycles = gp0cmds.cycles(self);
                        self.schedule_in(cycles, DmaTransportKind::Gpu(gp0cmds));
                    }
                }
                Ok(())
                // todo!("write at dma2chcr (gpu chcr): {chcr:#?}")
            }

            0x1f8010b0..=0x1f8010bf => todo!("write at dma3 (cdrom)"),
            0x1f8010c0..=0x1f8010cf => todo!("write at dma4 (spu)"),
            0x1f8010d0..=0x1f8010df => todo!("write at dma5 (pio)"),

            // dma 6
            0x1f8010e0 => {
                let madr = DmaMadr::new_with_raw_value(value.io_into_u32());
                self.dma_mut().dma6.madr = madr;
                tracing::trace!("write at dma6madr (otc madr): {:#?}", madr);
                Ok(())
            }
            0x1f8010e4 => {
                let bcr = DmaBcr::new_with_raw_value(value.io_into_u32());
                self.dma_mut().dma6.bcr = bcr;
                tracing::trace!("write at dma6bcr (otc bcr): {:#?}", bcr);
                Ok(())
            }
            0x1f8010e8 => {
                let chcr = DmaChcr::new_with_raw_value(value.io_into_u32());
                self.dma_mut().dma6.chcr = chcr;
                match self.dma().dma6.chcr.transfer() {
                    Transfer::StoppedCompleted => {}
                    Transfer::StartBusy => {
                        self.schedule_in(
                            self.dma().dma6.bcr.block_count() as u64,
                            DmaTransportKind::Otc(OTC),
                        );
                    }
                }
                tracing::trace!("write at dma6chcr (otc chcr): {:#?}", chcr);
                Ok(())
            }

            // dpcr
            0x1f8010f0 => {
                self.dma_mut().dpcr = Dpcr::new_with_raw_value(value.io_into_u32());
                Ok(())
            }
            // dicr
            0x1f8010f4 => {
                self.dma_mut().dicr = Dicr::new_with_raw_value(value.io_into_u32());
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn schedule_in(&mut self, cycles: u64, dma_kind: DmaTransportKind) {
        let finish_at = self.cpu().cycles + cycles;
        self.dma_mut()
            .queue
            .heap
            .push(DmaEvent {
                finish_at,
                dma_t: dma_kind,
            })
            .unwrap();
    }

    fn run_dma_transfers(&mut self) {
        if !self.dma().queue.heap.is_empty() {
            tracing::info!("polling {} dma events", self.dma().queue.heap.len());
        }
        while let Some(event) = self.dma().queue.heap.peek() {
            if event.finish_at > self.cpu().cycles {
                break;
            }
            match event.dma_t {
                DmaTransportKind::Otc(_) => {
                    OTC::write_data(self);
                    OTC::channel_mut(self)
                        .chcr
                        .set_transfer(Transfer::StoppedCompleted);
                }
                DmaTransportKind::Gpu(_) => {
                    GP0Cmds::write_data(self);
                    GP0Cmds::channel_mut(self)
                        .chcr
                        .set_transfer(Transfer::StoppedCompleted);
                }
            }
            _ = self.dma_mut().queue.heap.pop();
        }
    }
}

impl Dma for Emu {
    fn dma(&self) -> &DmaState {
        &self.dma
    }

    fn dma_mut(&mut self) -> &mut DmaState {
        &mut self.dma
    }
}

/// ## 1F8010F0h - DPCR - DMA Control Register (R/W)
///
///  0-2   DMA0, MDECin  Priority      (0..7; 0=Highest, 7=Lowest)
///  3     DMA0, MDECin  Master Enable (0=Disable, 1=Enable)
///  4-6   DMA1, MDECout Priority      (0..7; 0=Highest, 7=Lowest)
///  7     DMA1, MDECout Master Enable (0=Disable, 1=Enable)
///  8-10  DMA2, GPU     Priority      (0..7; 0=Highest, 7=Lowest)
///  11    DMA2, GPU     Master Enable (0=Disable, 1=Enable)
///  12-14 DMA3, CDROM   Priority      (0..7; 0=Highest, 7=Lowest)
///  15    DMA3, CDROM   Master Enable (0=Disable, 1=Enable)
///  16-18 DMA4, SPU     Priority      (0..7; 0=Highest, 7=Lowest)
///  19    DMA4, SPU     Master Enable (0=Disable, 1=Enable)
///  20-22 DMA5, PIO     Priority      (0..7; 0=Highest, 7=Lowest)
///  23    DMA5, PIO     Master Enable (0=Disable, 1=Enable)
///  24-26 DMA6, OTC     Priority      (0..7; 0=Highest, 7=Lowest)
///  27    DMA6, OTC     Master Enable (0=Disable, 1=Enable)
///  28-30 CPU memory access priority  (0..7; 0=Highest, 7=Lowest)
///  31    No effect, should be CPU memory access enable (R/W)
#[bitfield(u32)]
#[derive(Debug, Default)]
pub struct Dpcr {
    #[bits(0..=2, rw)]
    dma0prio: u3,
    #[bit(3, rw)]
    dma0on:   bool,

    #[bits(4..=6, rw)]
    dma1prio: u3,
    #[bit(7, rw)]
    dma1on:   bool,

    #[bits(8..=10, rw)]
    dma2prio: u3,
    #[bit(11, rw)]
    dma2on:   bool,

    #[bits(12..=14, rw)]
    dma3prio: u3,
    #[bit(15, rw)]
    dma3on:   bool,

    #[bits(16..=18, rw)]
    dma4prio: u3,
    #[bit(19, rw)]
    dma4on:   bool,

    #[bits(20..=22, rw)]
    dma5prio: u3,
    #[bit(23, rw)]
    dma5on:   bool,

    #[bits(24..=26, rw)]
    dma6prio: u3,
    #[bit(27, rw)]
    dma6on:   bool,

    #[bits(28..=30, rw)]
    cpu_prio: u3,
    /// no effect
    #[bit(31, rw)]
    _cpu_on:  bool,
}

/// # 1F8010F4h - DICR - DMA Interrupt Register (R/W)
///
///   0-6   Controls channel 0-6 completion interrupts in bits 24-30.
///         When 0, an interrupt only occurs when the entire transfer completes.
///         When 1, interrupts can occur for every slice and linked-list transfer.
///         No effect if the interrupt is masked by bits 16-22.
///   7-14  Unused
///   15    Bus error flag. Raised when transferring to/from an address outside of RAM. Forces bit 31. (R/W)
///   16-22 Channel 0-6 interrupt mask. If enabled, channels cause interrupts as per bits 0-6.
///   23    Master channel interrupt enable.
///   24-30 Channel 0-6 interrupt flags. (R, write 1 to reset)
///   31    Master interrupt flag (R)
#[bitfield(u32)]
#[derive(Debug, Default)]
pub struct Dicr {
    #[bit(0, rw)]
    irq_mode:   [DmaIrqMode; 7],
    #[bits(7..=14)]
    _padding:   u8,
    #[bit(15)]
    bus_error:  bool,
    #[bit(16, rw)]
    irq_mask:   [bool; 7],
    #[bit(23)]
    master_on:  bool,
    #[bit(24, rw)]
    master_on:  [bool; 7],
    #[bit(31)]
    master_irq: bool,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum DmaIrqMode {
    OnComplete,
    OnChunk,
}

/// ## 1F801080h+N*10h - D#_MADR - DMA base address (Channel 0..6) (R/W)
///
///   0-23  Memory Address where the DMA will start reading from/writing to
///   24-31 Not used (always zero)
#[bitfield(u32, debug)]
#[derive(Default, PartialEq, Eq)]
pub struct DmaMadr {
    #[bits(0..=23, rw)]
    addr: u24,
}

/// ## 1F801084h+N*10h - D#_BCR - DMA Block Control (Channel 0..6) (R/W)
///
/// For SyncMode=0 (ie. for OTC and CDROM):
///
///   0-15  BC    Number of words (0001h..FFFFh) (or 0=10000h words)
///   16-31 0     Not used (usually 0 for OTC, or 1 ("one block") for CDROM)
///
/// For SyncMode=1 (ie. for MDEC, SPU, and GPU-vram-data):
///
///   0-15  BS    Blocksize (words) ;for GPU/SPU max 10h, for MDEC max 20h
///   16-31 BA    Amount of blocks  ;ie. total length = BS*BA words
///
/// For SyncMode=2 (ie. for GPU-command-lists):
///
///   0-31  0     Not used (should be zero) (transfer ends at END-CODE in list)
#[bitfield(u32, debug)]
#[derive(Default, PartialEq, Eq)]
pub struct DmaBcr {
    // s0
    #[bits(0..=15, rw)]
    block_count: u16,

    // s1
    #[bits(0..=15, rw)]
    block_size:   u16,
    #[bits(16..=31, rw)]
    block_amount: u16,
}

/// ## 1F801088h+N*10h - D#_CHCR - DMA Channel Control (Channel 0..6) (R/W)
///
/// ```md
///   0     Transfer direction (0=device to RAM, 1=RAM to device)
///   1     MADR increment per step (0=+4, 1=-4)
///   2-7   Unused
///   8     When 1:
///         -Burst mode: enable "chopping" (cycle stealing by CPU)
///         -Slice mode: Causes DMA to hang
///         -Linked-list mode: Transfer header before data?
///   9-10  Transfer mode (SyncMode)
///         0=Burst (transfer data all at once after DREQ is first asserted)
///         1=Slice (split data into blocks, transfer next block whenever DREQ is asserted)
///         2=Linked-list mode
///         3=Reserved
///   11-15 Unused
///   16-18 Chopping DMA window size (1 << N words)
///   19    Unused
///   20-22 Chopping CPU window size (1 << N cycles)
///   23    Unused
///   24    Start transfer (0=stopped/completed, 1=start/busy)
///   25-27 Unused
///   28    Force transfer start without waiting for DREQ
///   29    In forced-burst mode, pauses transfer while set.
///         In other modes, stops bit 28 from being cleared after a slice is transferred.
///         No effect when transfer was caused by a DREQ.
///   30    Perform bus snooping (allows DMA to read from -nonexistent- cache?)
///   31    Unused
/// ```
#[bitfield(u32, debug)]
#[derive(Default, PartialEq, Eq)]
pub struct DmaChcr {
    #[bit(0, rw)]
    direction:     TransferDir,
    #[bit(1, rw)]
    madr_inc:      MadrInc,
    // unused: 2..=7
    #[bit(8, rw)]
    meta:          bool,
    #[bits(9..=10, rw)]
    sync_mode:     SyncMode,
    // unused: 11..=15
    #[bits(16..=18, rw)]
    chop_dma_size: u3,
    // unused: 19
    #[bits(20..=22, rw)]
    chop_cpu_size: u3,
    // unused: 23
    #[bit(24, rw)]
    transfer:      Transfer,
    // unused: 25..=27
    #[bit(28, rw)]
    force_start:   bool,

    #[bit(29, rw)]
    force_burst_lock: bool,

    #[bit(30, rw)]
    bus_snooping: bool,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug, PartialEq, PartialOrd, Ord, Eq)]
enum TransferDir {
    DeviceToRam = 0x0,
    RamToDevice = 0x1,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum MadrInc {
    Positive = 0x0,
    Negative = 0x1,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug, PartialEq, Eq)]
pub enum SyncMode {
    Burst      = 0x0,
    Slice      = 0x1,
    LinkedList = 0x2,
    Reserved   = 0x3,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum Transfer {
    StoppedCompleted = 0x0,
    StartBusy        = 0x1,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DmaChannel {
    pub madr: DmaMadr,
    pub bcr:  DmaBcr,
    pub chcr: DmaChcr,
}

impl DmaChannel {
    fn transfer(&self) -> Transfer {
        self.chcr.transfer()
    }
    fn io_set_madr<T: Copy>(&mut self, value: T) {
        let madr = DmaMadr::new_with_raw_value(value.io_into_u32());
        self.madr = madr;
    }
    fn io_set_bcr<T: Copy>(&mut self, value: T) {
        let bcr = DmaBcr::new_with_raw_value(value.io_into_u32());
        self.bcr = bcr;
    }
}

#[derive(Debug, Default, Clone)]
pub struct DmaQueue {
    heap: heapless::BinaryHeap<DmaEvent, Min, 6>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaEvent {
    finish_at: u64,
    dma_t:     DmaTransportKind,
}

impl PartialOrd for DmaEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DmaEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.finish_at.cmp(&other.finish_at) {
            core::cmp::Ordering::Equal => self.dma_t.idx().cmp(&other.dma_t.idx()),
            ord => ord,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaTransportKind {
    Otc(OTC),
    Gpu(GP0Cmds),
}

impl DmaTransportKind {
    pub fn idx(&self) -> u8 {
        match self {
            DmaTransportKind::Otc(_) => 6,
            DmaTransportKind::Gpu(_) => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OTC;

impl<T: Dma + IO + Fastmem + ?Sized> DmaTransport<T> for OTC {
    fn channel_mut(emu: &mut T) -> &mut DmaChannel {
        &mut emu.dma_mut().dma6
    }
    fn write_data(emu: &mut T) {
        let channel = Self::channel_mut(emu);
        let block_count = channel.bcr.block_count() as u32;
        let end = channel.madr.addr().as_u32();
        let end_node = DmaNodeHeader::new_with_raw_value(0x0).with_next(DmaNodeHeader::END.as_());
        Fastmem::write(emu, end, end_node).expect("dma6 otc write must go to ram!");
        let mut prev = end;
        for _ in 0..block_count {
            Fastmem::write(
                emu,
                prev + 0x4,
                DmaNodeHeader::default().with_next(prev.as_()),
            )
            .expect("dma6 otc write must go to ram!");
            prev += 0x4;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GP0Cmds {
    init_chan: DmaChannel,
}

impl GP0Cmds {
    fn cycles(&self, emu: &(impl IO + Fastmem + ?Sized)) -> u64 {
        let direction = self.init_chan.chcr.direction();
        let sync_mode = self.init_chan.chcr.sync_mode();
        debug_assert_eq!(direction, TransferDir::RamToDevice);
        debug_assert_eq!(sync_mode, SyncMode::LinkedList);
        let mut addr = self.init_chan.madr.addr().as_u32();
        let mut count = 0;
        loop {
            if count >= 1024 {
                panic!("infinite loop detected")
            }
            let header = Fastmem::read::<DmaNodeHeader>(emu, addr).unwrap();
            if header.is_end_marker() {
                break;
            }
            addr = header.next().as_u32();
            count += 1;
        }
        count
    }
}

impl<T: IO + Dma + ?Sized> DmaTransport<T> for GP0Cmds {
    fn channel_mut(emu: &mut T) -> &mut DmaChannel {
        &mut emu.dma_mut().dma2
    }
    fn write_data(emu: &mut T) {
        let channel = Self::channel_mut(emu);
        let direction = channel.chcr.direction();
        let sync_mode = channel.chcr.sync_mode();
        debug_assert_eq!(direction, TransferDir::RamToDevice);
        debug_assert_eq!(sync_mode, SyncMode::LinkedList);
        let mut addr = channel.madr.addr().as_u32();
        let mut count = 0;
        loop {
            if count >= 2048 {
                panic!("infinite loop detected");
            }
            let header = Fastmem::read::<DmaNodeHeader>(emu, addr).unwrap();
            let len = header.len();
            for idx in 0..len {
                let cmd = Fastmem::read::<u32>(emu, addr + idx as u32 * 0x4 + 0x4).unwrap();
                _ = emu.gpu_mut().gp0cmd_queue.push_back(cmd);
            }
            addr = header.next().as_u32();
            count += 1;
            if header.is_end_marker() {
                break;
            }
        }
    }
}

#[bitfield(u32, debug)]
#[derive(Default)]
pub struct DmaNodeHeader {
    #[bits(0..=23,rw)]
    next: u24,
    #[bits(24..=31,rw)]
    len:  u8,
}

impl DmaNodeHeader {
    pub const END: u32 = 0x00ff_ffff;
    fn is_end_marker(&self) -> bool {
        match self.next().value() {
            Self::END => true,
            value if value & 0x0040_0000 != 0 => true,
            _ => false,
        }
    }
}

pub trait DmaTransport<T: IO + ?Sized> {
    fn channel_mut(emu: &mut T) -> &mut DmaChannel;
    #[allow(unused_variables)]
    fn write(emu: &mut T, addr: u32) {}
    fn write_data(emu: &mut T) {
        let channel = Self::channel_mut(emu);
        let direction = channel.chcr.madr_inc();
        let step = match direction {
            MadrInc::Positive => 4i32,
            MadrInc::Negative => -4i32,
        };
        match channel.chcr.sync_mode() {
            SyncMode::Burst => {
                let block_count = channel.bcr.block_count() as u32;
                let mut addr = channel.madr.addr().as_u32();
                let mut i = 0;
                while i < block_count {
                    Self::write(emu, addr);
                    i += 1;
                    addr = addr.wrapping_add_signed(step);
                }
            }
            SyncMode::Slice => todo!(),
            SyncMode::LinkedList => {
                let transfer_direction = channel.chcr.direction();
                debug_assert_eq!(transfer_direction, TransferDir::RamToDevice);
                let mut addr = channel.madr.addr().as_u32();

                loop {
                    let value = emu.read::<DmaNodeHeader>(addr);
                    if value.is_end_marker() {
                        break;
                    }
                    // Self::write(emu, value.raw_value());
                    let len = value.len();
                    for offset in 1..=len {
                        let new_addr = addr + offset as u32 * size_of::<u32>() as u32;
                        Self::write(emu, new_addr);
                    }
                    addr += len as u32 * 0x4;
                }
            }
            SyncMode::Reserved => {}
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::io::dma::Dicr;

    use super::DmaIrqMode;

    #[test]
    fn test_dicr_irq_mode() {
        let dicr = Dicr::default();
        let dicr = dicr.with_irq_mode(0, DmaIrqMode::OnComplete);
        let dicr = dicr.with_irq_mode(1, DmaIrqMode::OnComplete);
        let dicr = dicr.with_irq_mode(2, DmaIrqMode::OnChunk);
        let dicr = dicr.with_irq_mode(3, DmaIrqMode::OnChunk);
        let dicr = dicr.with_irq_mode(4, DmaIrqMode::OnComplete);
        let dicr = dicr.with_irq_mode(5, DmaIrqMode::OnChunk);
        let dicr = dicr.with_irq_mode(6, DmaIrqMode::OnComplete);
        assert_eq!(dicr.raw_value(), 0b00101100)
    }
}
