use crate::gpu::GpuCmd;
use crate::io::evque::EvCtx;
use crate::io::irq::Irq;
use crate::io::{CastIOFrom, CastIOInto, IOResult, UnhandledIO};
use crate::{Emu, trace_todo};
use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use pchan_macros::{pchan_instrument_read, pchan_instrument_write};
use pchan_utils::hex;
use slab::Slab;

#[derive(derive_more::Debug, Clone)]
pub struct DmaState {
    dpcr: Dpcr,
    dicr: Dicr,
    // TODO: dma channels
    dma2: DmaChannel,
    dma3: DmaChannel,
    dma6: DmaChannel,

    // events: Slab<DmaEvent>,
    ongoing_transfer: Option<DmaEvent>,

    #[debug(skip)]
    debug_cdrom_data: Vec<u8>,
}

impl Default for DmaState {
    fn default() -> Self {
        Self {
            dpcr:             Dpcr::new_with_raw_value(0x07654321),
            dicr:             Dicr::default(),
            dma2:             DmaChannel::default(),
            dma3:             DmaChannel::default(),
            dma6:             DmaChannel::default(),
            // queue: DmaQueue::default(),
            // events:           Slab::with_capacity(1024),
            debug_cdrom_data: vec![],
            ongoing_transfer: None,
        }
    }
}

impl DmaState {
    pub fn dump_cdrom_data(&self) {
        use std::io::Write;

        let mut f = std::fs::File::create("read_from_cdrom.bin").unwrap();
        _ = f.write(&self.debug_cdrom_data);
    }
}

///
/// # DMA Register Summary
///
/// ```plaintext
///   1F80108xh DMA0 channel 0  MDECin  (RAM to MDEC)
///   1F80109xh DMA1 channel 1  MDECout (MDEC to RAM)
///   1F8010Axh DMA2 channel 2  GPU (lists + image data)
///   1F8010Bxh DMA3 channel 3  CDROM   (CDROM to RAM)
///   1F8010Cxh DMA4 channel 4  SPU
///   1F8010Dxh DMA5 channel 5  PIO (Expansion Port)
///   1F8010Exh DMA6 channel 6  OTC (reverse clear OT) (GPU related)
///   1F8010F0h DPCR - DMA Control register
///   1F8010F4h DICR - DMA Interrupt register
/// ```
///
/// These ports control DMA at the CPU-side. In most cases, you'll additionally
/// need to initialize an address (and transfer direction, transfer enabled, etc.)
/// at the remote-side (eg. at the GPU-side for DMA2).
impl Emu {
    #[pchan_instrument_read("dma:r")]
    pub fn dma_read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801080..=0x1f80108f => trace_todo!(0x0, "read at dma0 (MDECin)"),
            0x1f801090..=0x1f80109f => trace_todo!(0x0, "read at dma1 (MDECout)"),

            // dma 2
            0x1f8010a0 => Ok(self.dma().dma2.madr.addr().io_from_u32()),
            0x1f8010a4 => Ok(self.dma().dma2.bcr.io_from_u32()),
            0x1f8010a8 => {
                let chcr = self.dma().dma2.chcr;
                tracing::trace!("read at dma2chcr (gpu chcr): {:?}", chcr.transfer());
                Ok(chcr.io_from_u32())
            }

            0x1f8010b0..=0x1f8010bf => trace_todo!(0x0, "read at dma3 (cdrom)"),
            0x1f8010c0..=0x1f8010cf => trace_todo!(0x0, "read at dma4 (spu)"),
            0x1f8010d0..=0x1f8010df => trace_todo!(0x0, "read at dma5 (pio)"),

            // dma 6
            0x1f8010e0 => trace_todo!(0x0, "read at dma6madr (otc madr)"),
            0x1f8010e4 => trace_todo!(0x0, "read at dma6bcr (otc bcr)"),
            0x1f8010e8 => {
                let chcr = self.dma().dma6.chcr;
                tracing::trace!(dma6 = ?chcr.transfer(), "read at dma6chcr (otc chcr)");
                Ok(chcr.io_from_u32())
            }

            0x1f8010f0 => Ok(self.dma().dpcr.io_from_u32()),
            0x1f8010f4 => Ok(self.dma().dicr.io_from_u32()),
            0x1f8010f8 => trace_todo!(0x0, "todo(dma): read at dma transfer complete register"),
            0x1f8010fc => trace_todo!(0x0, "todo(dma): read at dma otc fill value"),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[pchan_instrument_write("dma:w")]
    pub fn dma_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801080..=0x1f80108f => trace_todo!("write at dma0 (MDECin)"),
            0x1f801090..=0x1f80109f => trace_todo!("write at dma1 (MDECout)"),

            // dma 2
            0x1f8010a0 => Dma2Gpu::write_madr(self, value),
            0x1f8010a4 => Dma2Gpu::write_bcr(self, value),
            0x1f8010a8 => Dma2Gpu::write_chcr(self, value),

            // dma 3
            0x1f8010b0 => Dma3Cdrom::write_madr(self, value),
            0x1f8010b4 => Dma3Cdrom::write_bcr(self, value),
            0x1f8010b8 => Dma3Cdrom::write_chcr(self, value),

            // 0x1f8010b0..=0x1f8010bf => trace_todo!("write at dma3 (cdrom)"),
            0x1f8010c0..=0x1f8010cf => trace_todo!("write at dma4 (spu)"),
            0x1f8010d0..=0x1f8010df => trace_todo!("write at dma5 (pio)"),

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
                    self.dma_start_transfer(
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

                let old_irq_flags = irq_flags;
                // writing 1 to irq flag resets it to 0
                let irq_flags = irq_flags & !new_irq_flags;
                tracing::info!(
                    "write at dicr: dma_irq_flags: {old_irq_flags:08b} -> {irq_flags:08b}"
                );

                let new_dicr = new_dicr.with_combined_irq_flags(irq_flags);

                *dicr = new_dicr;
                self.update_dicr_master_irq_flag();
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn dma_start_transfer(&mut self, event: DmaEvent) {
        match self.dma.ongoing_transfer {
            Some(old) => {
                tracing::debug!("dma stall: already transfering");
                let clock = old.cycles(self);
                self.handle_dma_event(old);
                // self.cpu.d_clock = clock as u32;
                // self.run_io();
                self.dma.ongoing_transfer = None;

                become self.dma_start_transfer(event);
            }
            None => {
                tracing::debug!("dma: schedulde dma event: {:#?}", event);
                self.dma.ongoing_transfer = Some(event);
                self.evque_mut().schedule(
                    |emu, _| {
                        if let Some(transfer) = emu.dma.ongoing_transfer.take() {
                            emu.handle_dma_event(transfer);
                        }
                    },
                    0,
                    event.in_cycles,
                );
            }
        };
    }

    fn handle_dma_event(&mut self, event: DmaEvent) {
        tracing::debug!("dma proc");
        match event.dma_t {
            DmaTransportKind::Otc => {
                self.dma6_write_data(event);
            }
            DmaTransportKind::Gpu => {
                self.dma2_write_data(event);
            }
            DmaTransportKind::Cdrom => {
                self.dma3_write_data(event);
            }
        }
    }

    fn dma_irq_raise_complete(&mut self, idx: usize) {
        let dicr = &mut self.dma_mut().dicr;
        if dicr.irq_mask(idx) && dicr.master_chan_irq() {
            dicr.set_irq_flag(idx, true);
        }
        let old_master_irq = dicr.master_irq();
        self.update_dicr_master_irq_flag();
        let dicr = &mut self.dma_mut().dicr;
        if let (false, true) = (old_master_irq, dicr.master_irq()) {
            self.irq_trigger(Irq::Irq3Dma);
        }
    }

    fn update_dicr_master_irq_flag(&mut self) {
        let dicr = &mut self.dma_mut().dicr;
        let new_master_irq =
            dicr.bus_error() || (dicr.master_chan_irq() && dicr.combined_irq_flags().as_u8() > 0);
        dicr.set_master_irq(new_master_irq);
    }

    fn create_dma_event(&self, channel: DmaChannel, kind: DmaTransportKind) -> DmaEvent {
        match channel.chcr.sync_mode() {
            SyncMode::Burst => DmaEvent {
                in_cycles: channel.burst_cycles(kind),
                init_chan: channel,
                slice:     None,
                dma_t:     kind,
            },
            SyncMode::Slice => DmaEvent {
                in_cycles: channel.slice_cycles(),
                init_chan: channel,
                slice:     Some(SliceTransferState {
                    addr: channel.madr.addr().as_u32(),
                    idx:  0,
                }),
                dma_t:     kind,
            },
            SyncMode::LinkedList => DmaEvent {
                in_cycles: channel.linked_list_cycles(self),
                init_chan: channel,
                slice:     None,
                dma_t:     kind,
            },
            SyncMode::Reserved => DmaEvent {
                in_cycles: 0,
                init_chan: channel,
                slice:     None,
                dma_t:     kind,
            },
        }
    }

    fn dma_write_data<T: Transfer>(&mut self, event: DmaEvent, transfer: &mut T) {
        let init_chan = event.init_chan;
        let direction = init_chan.chcr.direction();
        let sync_mode = init_chan.chcr.sync_mode();
        let clock = self.cpu().cycles;
        let idx = event.dma_t.idx() as usize;
        match sync_mode {
            SyncMode::Slice => {
                let mut current_event = Some(event);
                while let Some(event) = current_event {
                    let slice = event
                        .slice
                        .expect("event with sync mode slice has no slice state. this is a bug.");
                    let mut addr = slice.addr;
                    T::channel(self).madr.set_addr(addr.as_());
                    let len = init_chan.bcr.s1_block_size();
                    for _ in 0..len {
                        match direction {
                            TransferDir::DeviceToRam => {
                                transfer.read(self, addr);
                            }
                            TransferDir::RamToDevice => {
                                transfer.write(self, addr);
                            }
                        }
                        addr += 0x4;
                    }

                    // do not mark as done until final event is reached
                    tracing::debug!(
                        "dma event.{} [{}/{}]",
                        hex(init_chan.madr.addr().as_u32()),
                        slice.idx,
                        init_chan.bcr.s1_block_count()
                    );

                    if slice.idx >= init_chan.bcr.s1_block_count() as u32 - 1 {
                        tracing::info!(
                            "dma event.{} finished",
                            hex(init_chan.madr.addr().as_u32())
                        );

                        T::channel(self).set_complete();
                        self.dma_irq_raise_complete(idx);

                        break;
                    } else {
                        let bcr = &mut T::channel(self).bcr;
                        bcr.set_s1_block_count(bcr.s1_block_count() - 1);

                        if let DmaIrqMode::OnChunk = self.dma.dicr.irq_mode(idx) {
                            T::channel(self).set_complete();
                            self.dma_irq_raise_complete(idx);
                        }

                        let cycles_per_step = event.init_chan.slice_cycles();
                        let addr_step = event.init_chan.bcr.s1_block_size();
                        let upcoming = clock + cycles_per_step;
                        let slice = SliceTransferState {
                            addr: slice.addr + addr_step as u32 * 0x4,
                            idx:  slice.idx + 1,
                        };
                        let next_event = DmaEvent {
                            in_cycles: cycles_per_step,
                            init_chan: event.init_chan,
                            slice:     Some(slice),
                            dma_t:     event.dma_t,
                        };

                        if upcoming < clock {
                            current_event = Some(next_event);
                        } else {
                            self.dma_start_transfer(next_event);

                            break;
                        }
                    }
                }
            }
            SyncMode::Burst => {
                let mut addr = init_chan.madr.addr().as_u32();
                for _ in 0..init_chan.bcr.s0_word_count() {
                    match direction {
                        TransferDir::DeviceToRam => {
                            transfer.read(self, addr);
                        }
                        TransferDir::RamToDevice => {
                            transfer.write(self, addr);
                        }
                    }
                    addr += 0x4;
                }
                T::channel(self).set_complete();
                self.dma_irq_raise_complete(idx);
            }
            SyncMode::LinkedList => {
                let mut addr = init_chan.madr.addr().as_u32();
                let mut visited = heapless::index_set::FnvIndexSet::<u32, 2048>::new();
                let mut count = 0;
                loop {
                    if count >= 1024 + 128 {
                        panic!(
                            "infinite loop detected, dma n: {init_chan:#?}\ndpcr: {:#?}",
                            self.dma().dpcr
                        );
                    }
                    let header = self.read::<DmaNodeHeader>(addr);
                    tracing::trace!(header.next = %hex(header.next()), header.len = header.len());
                    let len = header.len();
                    for idx in 0..len {
                        let addr = addr + idx as u32 * 0x4 + 0x4;
                        transfer.write(self, addr);
                    }
                    visited.insert(addr).expect(
                        "bug: dma linked list traversal visited set capacity is too small.",
                    );
                    addr = header.next().as_u32();

                    // TODO: cycle detection

                    count += 1;
                    if header.is_end_marker() {
                        T::channel(self).madr.set_addr(DmaNodeHeader::END.as_());
                        break;
                    }
                }
                T::channel(self).set_complete();
                self.dma_irq_raise_complete(idx);
                tracing::trace!("end gp0 linked list traversal");
            }
            SyncMode::Reserved => todo!(),
        }
    }

    fn dma2_write_data(&mut self, event: DmaEvent) {
        self.dma_write_data(event, &mut Dma2Gpu);
    }

    fn dma3_write_data(&mut self, event: DmaEvent) {
        self.dma_write_data(event, &mut Dma3Cdrom);
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
            self.fastmem_write(addr, node)
                .expect("dma6 otc write must go to ram!");
            addr = next_addr;
        }

        let end_node = DmaNodeHeader::new_with_raw_value(DmaNodeHeader::END);
        self.fastmem_write(addr, end_node)
            .expect("dma6 otc write must go to ram!");
        self.dma_mut().dma6.set_complete();
    }
}

trait Transfer {
    const TRANSPORT_KIND: DmaTransportKind;

    /// ram to device
    fn write(&mut self, emu: &mut Emu, address: u32);
    /// device to ram
    fn read(&mut self, emu: &mut Emu, address: u32);
    fn channel(emu: &mut Emu) -> &mut DmaChannel;

    fn write_madr<T: Copy>(emu: &mut Emu, value: T) -> IOResult<()> {
        Self::channel(emu).io_set_madr(value);
        Ok(())
    }

    fn write_bcr<T: Copy>(emu: &mut Emu, value: T) -> IOResult<()> {
        Self::channel(emu).io_set_bcr(value);
        Ok(())
    }

    fn write_chcr<T: Copy>(emu: &mut Emu, value: T) -> IOResult<()> {
        let dma = Self::channel(emu);
        let chcr = DmaChcr::new_with_raw_value(value.io_into_u32());
        dma.chcr = chcr;
        match dma.chcr.transfer() {
            TransferState::StoppedCompleted => {}
            TransferState::StartBusy => {
                let dma = *dma;
                emu.dma_start_transfer(emu.create_dma_event(dma, Self::TRANSPORT_KIND));
            }
        };
        Ok(())
    }
}

struct Dma2Gpu;

impl Transfer for Dma2Gpu {
    const TRANSPORT_KIND: DmaTransportKind = DmaTransportKind::Gpu;

    fn write(&mut self, emu: &mut Emu, address: u32) {
        let cmd = emu
            .fastmem_read::<GpuCmd>(address)
            .expect("address outside of ram/bios");
        emu.gpu_gp0_cmd(cmd);
    }

    fn read(&mut self, emu: &mut Emu, address: u32) {
        let value = emu.gpu_read::<u32>(0x1f801810).unwrap();
        _ = emu.fastmem_write(address, value);
    }

    fn channel(emu: &mut Emu) -> &mut DmaChannel {
        &mut emu.dma.dma2
    }
}

struct Dma3Cdrom;

impl Transfer for Dma3Cdrom {
    const TRANSPORT_KIND: DmaTransportKind = DmaTransportKind::Cdrom;

    fn write(&mut self, _emu: &mut Emu, _address: u32) {
        todo!()
    }

    fn read(&mut self, emu: &mut Emu, address: u32) {
        let value = emu.cdrom_read_data::<4>();
        emu.dma.debug_cdrom_data.extend_from_slice(&value);
        let value = u32::from_le_bytes(value);
        _ = emu.fastmem_write(address, value);
        tracing::debug!("copied byte {} to memory at {}", hex(value), hex(address));
    }

    fn channel(emu: &mut Emu) -> &mut DmaChannel {
        &mut emu.dma.dma3
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
    irq_mode:        [DmaIrqMode; 7],
    #[bits(7..=14)]
    _padding:        u8,
    #[bit(15, rw)]
    bus_error:       bool,
    #[bit(16, rw)]
    irq_mask:        [bool; 7],
    #[bit(23, rw)]
    master_chan_irq: bool,
    #[bit(24, rw)]
    irq_flag:        [bool; 7],

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
    transfer:      TransferState,
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
pub enum TransferState {
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
    fn transfer(&self) -> TransferState {
        self.chcr.transfer()
    }
    fn io_set_madr<T: Copy>(&mut self, value: T) {
        self.madr.set_addr(value.io_into_u32().as_());
    }
    fn io_set_bcr<T: Copy>(&mut self, value: T) {
        let bcr = DmaBcr::new_with_raw_value(value.io_into_u32());
        self.bcr = bcr;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaEvent {
    in_cycles: u64,
    init_chan: DmaChannel,
    slice:     Option<SliceTransferState>,
    dma_t:     DmaTransportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaTransportKind {
    Otc,
    Gpu,
    Cdrom,
}

impl DmaTransportKind {
    pub fn idx(&self) -> u8 {
        match self {
            DmaTransportKind::Otc => 6,
            DmaTransportKind::Gpu => 2,
            DmaTransportKind::Cdrom => 3,
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
    fn cycles(&self, emu: &Emu) -> u64 {
        let sync_mode = self.init_chan.chcr.sync_mode();
        match sync_mode {
            SyncMode::Burst => self.init_chan.burst_cycles(self.dma_t),
            SyncMode::Slice => self.init_chan.slice_cycles(),
            SyncMode::LinkedList => self.init_chan.linked_list_cycles(emu),
            SyncMode::Reserved => u64::MAX,
        }
    }
}

impl DmaChannel {
    fn linked_list_cycles(&self, emu: &Emu) -> u64 {
        let mut addr = self.madr.addr().as_u32();
        let mut visited = heapless::index_set::FnvIndexSet::<u32, 2048>::new();
        let mut count = 0;
        loop {
            let header = emu.fastmem_read::<DmaNodeHeader>(addr).unwrap();
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

    fn burst_cycles(&self, dma_t: DmaTransportKind) -> u64 {
        match dma_t {
            DmaTransportKind::Otc | DmaTransportKind::Gpu => self.bcr.s0_word_count() as u64,
            DmaTransportKind::Cdrom => self.bcr.s0_word_count() as u64 * 24,
        }
    }

    fn set_complete(&mut self) {
        self.chcr.set_transfer(TransferState::StoppedCompleted);
    }
}

#[expect(clippy::len_without_is_empty)]
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
