use crate::{
    Bus, Emu,
    gpu::Gpu,
    io::{CastIOFrom, CastIOInto, IO, IOResult, Interrupts, UnhandledIO, irq::Irq},
    memory::fastmem::Fastmem,
};
use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use heapless::binary_heap::Min;
use pchan_macros::{pchan_instrument_read, pchan_instrument_write};
use pchan_utils::hex;

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
pub trait Dma: Bus + IO + Fastmem + Interrupts + Gpu {
    #[pchan_instrument_read("dma:r")]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801080..=0x1f80108f => todo!("read at dma0 (MDECin)"),
            0x1f801090..=0x1f80109f => todo!("read at dma1 (MDECout)"),

            // dma 2
            0x1f8010a0 => Ok(self.dma().dma2.madr.addr().io_from_u32()),
            0x1f8010a4 => todo!("read at dma2bcr (gpu bcr)"),
            0x1f8010a8 => {
                let chcr = self.dma().dma2.chcr;
                tracing::trace!("read at dma2chcr (gpu chcr): {:?}", chcr.transfer());
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
                tracing::trace!(dma6 = ?chcr.transfer(), "read at dma6chcr (otc chcr)");
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
                tracing::trace!("write at dma2bcr (gpu bcr): {:#?}", self.dma().dma2.bcr);
                Ok(())
            }
            0x1f8010a8 => {
                let chcr = DmaChcr::new_with_raw_value(value.io_into_u32());
                self.dma_mut().dma2.chcr = chcr;
                tracing::trace!("write at dma2chcr (gpu chcr): {:#?}", chcr);
                match self.dma().dma2.chcr.transfer() {
                    Transfer::StoppedCompleted => {}
                    Transfer::StartBusy => {
                        self.dma_schedule(
                            self.create_dma_event(self.dma().dma2, DmaTransportKind::Gpu),
                        );
                    }
                }
                Ok(())
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

                tracing::trace!("write at dma6chcr (otc chcr): {:#?}", chcr);

                if chcr.raw_value() == 0x11000002 {
                    self.dma_schedule(
                        self.create_dma_event(self.dma().dma6, DmaTransportKind::Otc),
                    );
                    tracing::trace!("dma6 scheduled");
                }
                Ok(())
            }

            // dpcr
            0x1f8010f0 => {
                self.dma_mut().dpcr = Dpcr::new_with_raw_value(value.io_into_u32());
                Ok(())
            }
            // dicr
            0x1f8010f4 => {
                let dicr = &mut self.dma_mut().dicr;
                let new_dicr = Dicr::new_with_raw_value(value.io_into_u32());
                let irq_flags = dicr.combined_irq_flags();
                let new_irq_flags = new_dicr.combined_irq_flags();

                // writing 1 to irq flag resets it to 0
                let irq_flags = irq_flags & !new_irq_flags;
                tracing::trace!("read at dicr: dma_irq_flags: {irq_flags:b}");

                let new_dicr = new_dicr.with_combined_irq_flags(irq_flags);

                *dicr = new_dicr;
                self.update_dicr_master_irq_flag();
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn dma_schedule(&mut self, event: DmaEvent) {
        if let Some(slice) = event.slice {
            let cycles_per_step = event.init_chan.slice_cycles();
            let mut upcoming = event.upcoming;

            let start = slice.idx as u16;
            let end = event.init_chan.bcr.s1_block_count();
            let mut addr = event.init_chan.madr.addr().as_u32();
            let addr_step = event.init_chan.bcr.s1_block_size();
            for i in start..end {
                let slice = SliceTransferState {
                    addr,
                    idx: i as u32,
                };
                addr += addr_step as u32 * 0x4;
                self.dma_mut()
                    .queue
                    .heap
                    .push(DmaEvent {
                        upcoming,
                        init_chan: event.init_chan,
                        slice: Some(slice),
                        dma_t: event.dma_t,
                    })
                    .unwrap();
                upcoming += cycles_per_step;
            }
        } else {
            self.dma_mut().queue.heap.push(event).unwrap();
        }
    }

    fn run_dma_transfers(&mut self) {
        while let Some(event) = self.dma().queue.heap.peek() {
            if event.upcoming > self.cpu().cycles {
                break;
            }
            let idx = event.dma_t.idx();
            match event.dma_t {
                DmaTransportKind::Otc => {
                    self.dma6_write_data(*event);
                }
                DmaTransportKind::Gpu => {
                    self.dma2_write_data(*event);
                }
            }
            self.dma_irq_raise_complete(idx as usize);
            _ = self.dma_mut().queue.heap.pop();
        }
    }

    fn dma_irq_raise_complete(&mut self, idx: usize) {
        let dicr = &mut self.dma_mut().dicr;
        if dicr.irq_mask(idx) && dicr.master_on() {
            dicr.set_irq_flag(idx, true);
        }
        let old_master_irq = dicr.master_irq();
        self.update_dicr_master_irq_flag();
        let dicr = &mut self.dma_mut().dicr;
        if let (false, true) = (old_master_irq, dicr.master_irq()) {
            self.trigger_irq(Irq::Irq3Dma);
        }
    }

    fn update_dicr_master_irq_flag(&mut self) {
        let dicr = &mut self.dma_mut().dicr;
        let new_master_irq =
            dicr.bus_error() || (dicr.master_on() && dicr.combined_irq_flags().as_u8() > 0);
        dicr.set_master_irq(new_master_irq);
    }

    fn create_dma_event(&self, channel: DmaChannel, kind: DmaTransportKind) -> DmaEvent {
        let clock = self.cpu().cycles;
        match channel.chcr.sync_mode() {
            SyncMode::Burst => DmaEvent {
                upcoming:  clock + channel.burst_cycles(),
                init_chan: channel,
                slice:     None,
                dma_t:     kind,
            },
            SyncMode::Slice => DmaEvent {
                upcoming:  clock + channel.slice_cycles(),
                init_chan: channel,
                slice:     Some(SliceTransferState {
                    addr: channel.madr.addr().as_u32(),
                    idx:  0,
                }),
                dma_t:     kind,
            },
            SyncMode::LinkedList => DmaEvent {
                upcoming:  clock + channel.linked_list_cycles(self),
                init_chan: channel,
                slice:     None,
                dma_t:     kind,
            },
            SyncMode::Reserved => DmaEvent {
                upcoming:  0,
                init_chan: channel,
                slice:     None,
                dma_t:     kind,
            },
        }
    }

    fn dma2_write_data(&mut self, event: DmaEvent) {
        let channel = event.init_chan;
        let direction = channel.chcr.direction();
        let sync_mode = channel.chcr.sync_mode();
        match sync_mode {
            SyncMode::Slice => match direction {
                TransferDir::DeviceToRam => todo!(),
                TransferDir::RamToDevice => {
                    let slice = event
                        .slice
                        .expect("event with sync mode slice has no slice state. this is a bug.");

                    let mut addr = slice.addr;
                    let len = channel.bcr.s1_block_size();
                    tracing::trace!("dma: transfering slice of {} words...", len);
                    tracing::trace!(
                        "dma: current gp0 fifo capacity: {}/{}",
                        self.gpu().gp0cmd_queue.len(),
                        self.gpu().gp0cmd_queue.capacity()
                    );
                    for _ in 0..len {
                        let value = Fastmem::read(self, addr).unwrap();
                        // flushing the queue here is not ideal, as real dma
                        // would hang until the gpu has capacity for more
                        // commands. but our commands do not take actual
                        // time to execute
                        self.gp0_cmd_queue_push_or_flush(value);
                        addr += 0x4;
                    }
                    // do not mark as done until final event is reached
                    if slice.idx == channel.bcr.s1_block_count() as u32 - 1 {
                        self.dma_mut().dma2.set_complete();
                    }
                }
            },
            SyncMode::Burst => match direction {
                TransferDir::DeviceToRam => todo!(),
                TransferDir::RamToDevice => {
                    let mut addr = channel.madr.addr().as_u32();
                    for _ in 0..channel.bcr.s0_word_count() {
                        let value = Fastmem::read(self, addr).unwrap();
                        self.gp0_cmd_queue_push_or_flush(value);
                        addr += 0x4;
                    }
                    self.dma_mut().dma2.set_complete();
                }
            },
            SyncMode::LinkedList => {
                let mut addr = channel.madr.addr().as_u32();
                let mut visited = heapless::index_set::FnvIndexSet::<u32, 2048>::new();
                let mut count = 0;
                tracing::trace!("start gp0 linked list traversal");
                loop {
                    if count >= 1024 + 128 {
                        panic!(
                            "infinite loop detected, dma 2: {channel:#?}\ndpcr: {:#?}",
                            self.dma().dpcr
                        );
                    }
                    let header = IO::read::<DmaNodeHeader>(self, addr);
                    tracing::info!(header.next = %hex(header.next()), header.len = header.len());
                    let len = header.len();
                    for idx in 0..len {
                        let addr = addr + idx as u32 * 0x4 + 0x4;
                        let cmd = IO::read::<u32>(self, addr);
                        tracing::trace!(addr = %hex(addr), "dma2 push: {}", hex(cmd),);
                        self.gp0_cmd_queue_push_or_flush(cmd);
                    }
                    visited.insert(addr).expect(
                        "bug: dma2 linked list traversal visited set capacity is too small.",
                    );
                    addr = header.next().as_u32();

                    // // cycle detected, reschedule later
                    // //
                    // // rescheduling here adds a backpressure so the cpu has the chance to execute
                    // // and break chains in the cycle
                    // //
                    // // we might want to allow the chain to run for a few more cycles
                    // if visited.contains(&addr) {
                    //     let mut channel = channel;
                    //     channel.madr.set_addr(u24::new(addr));
                    //     self.dma_schedule(self.create_dma_event(channel, DmaTransportKind::Gpu));
                    //     return;
                    // }

                    count += 1;
                    if header.is_end_marker() {
                        break;
                    }
                }
                self.dma_mut().dma2.set_complete();
                tracing::trace!("end gp0 linked list traversal");
            }
            SyncMode::Reserved => todo!(),
        }
    }

    fn dma6_write_data(&mut self, event: DmaEvent) {
        let channel = event.init_chan;
        let mut word_count = channel.bcr.s0_word_count() as u32;
        tracing::trace!("dma6 start write:\n{:#?}", channel);

        if word_count == 0 {
            word_count = 0x10000;
        }

        let start = channel.madr.addr().as_u32();

        let mut addr = start;
        // end node is written separately
        for _ in 0..(word_count - 1) {
            let next_addr = addr - 0x4;
            let node = DmaNodeHeader::default().with_next(next_addr.as_());
            Fastmem::write(self, addr, node).expect("dma6 otc write must go to ram!");
            addr = next_addr;
        }

        let end_node = DmaNodeHeader::new_with_raw_value(DmaNodeHeader::END);
        Fastmem::write(self, addr, end_node).expect("dma6 otc write must go to ram!");
        self.dma_mut().dma6.set_complete();
    }
}

impl DmaState {
    pub fn pending_event(&self) -> Option<u64> {
        self.queue.heap.peek().map(|event| event.upcoming + 0x1)
    }
}

impl Dma for Emu {}

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
#[bitfield(u32, debug)]
#[derive(Default)]
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
    irq_mode:  [DmaIrqMode; 7],
    #[bits(7..=14)]
    _padding:  u8,
    #[bit(15, rw)]
    bus_error: bool,
    #[bit(16, rw)]
    irq_mask:  [bool; 7],
    #[bit(23, rw)]
    master_on: bool,
    #[bit(24, rw)]
    irq_flag:  [bool; 7],

    #[bits(24..=30, rw)]
    combined_irq_flags: u7,

    #[bit(31, rw)]
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
    s0_word_count: u16,

    // s1
    #[bits(0..=15, rw)]
    s1_block_size:  u16,
    #[bits(16..=31, rw)]
    s1_block_count: u16,
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
    heap: Box<heapless::BinaryHeap<DmaEvent, Min, 128>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaEvent {
    upcoming:  u64,
    init_chan: DmaChannel,
    slice:     Option<SliceTransferState>,
    dma_t:     DmaTransportKind,
}

impl PartialOrd for DmaEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DmaEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.upcoming.cmp(&other.upcoming) {
            core::cmp::Ordering::Equal => self.dma_t.idx().cmp(&other.dma_t.idx()),
            ord => ord,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaTransportKind {
    Otc,
    Gpu,
}

impl DmaTransportKind {
    pub fn idx(&self) -> u8 {
        match self {
            DmaTransportKind::Otc => 6,
            DmaTransportKind::Gpu => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OTC;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SliceTransferState {
    addr: u32,
    idx:  u32,
}

impl DmaEvent {
    fn cycles(&self, emu: &(impl IO + Fastmem + ?Sized)) -> u64 {
        let sync_mode = self.init_chan.chcr.sync_mode();
        match sync_mode {
            SyncMode::Burst => self.init_chan.burst_cycles(),
            SyncMode::Slice => self.init_chan.slice_cycles(),
            SyncMode::LinkedList => self.init_chan.linked_list_cycles(emu),
            SyncMode::Reserved => u64::MAX,
        }
    }
}

impl DmaChannel {
    fn linked_list_cycles(&self, emu: &(impl Fastmem + ?Sized)) -> u64 {
        let mut addr = self.madr.addr().as_u32();
        let mut visited = heapless::index_set::FnvIndexSet::<u32, 2048>::new();
        let mut count = 0;
        loop {
            let header = Fastmem::read::<DmaNodeHeader>(emu, addr).unwrap();
            visited
                .insert(addr)
                .expect("bug: visited set capacity is too small. consider increasing or use heap.");
            addr = header.next().as_u32();
            if visited.contains(&addr) {
                // cycle detected, return early and reschedule later
                return count;
            }
            count += header.len() as u64;
            if header.is_end_marker() {
                break;
            }
        }
        count
    }

    fn slice_cycles(&self) -> u64 {
        self.bcr.s1_block_size() as u64
    }

    fn burst_cycles(&self) -> u64 {
        self.bcr.s0_word_count() as u64
    }

    fn set_complete(&mut self) {
        self.chcr.set_transfer(Transfer::StoppedCompleted);
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
        self.next().value() == Self::END
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
