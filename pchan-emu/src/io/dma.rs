use crate::{
    Bus, Emu,
    io::{CastIOFrom, CastIOInto, IOResult, UnhandledIO},
};
use arbitrary_int::*;
use bitbybit::{bitenum, bitfield};
use pchan_macros::{pchan_instrument_read, pchan_instrument_write};

#[derive(Debug, Clone)]
pub struct DmaState {
    dpcr: Dpcr,
    dicr: Dicr,
}

impl Default for DmaState {
    fn default() -> Self {
        Self {
            dpcr: Dpcr::new_with_raw_value(0x07654321),
            dicr: Dicr::default(),
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
pub trait Dma: Bus {
    fn dma(&self) -> &DmaState;
    fn dma_mut(&mut self) -> &mut DmaState;
    #[pchan_instrument_read("dma:r")]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801080..=0x1f80108f => todo!("read at dma0 (MDECin)"),
            0x1f801090..=0x1f80109f => todo!("read at dma1 (MDECout)"),
            0x1f8010a0..=0x1f8010af => todo!("read at dma2 (gpu)"),
            0x1f8010b0..=0x1f8010bf => todo!("read at dma3 (cdrom)"),
            0x1f8010c0..=0x1f8010cf => todo!("read at dma4 (spu)"),
            0x1f8010d0..=0x1f8010df => todo!("read at dma5 (pio)"),
            0x1f8010e0..=0x1f8010ef => todo!("read at dma6 (otc)"),
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
            0x1f8010a0..=0x1f8010af => todo!("write at dma2 (gpu)"),
            0x1f8010b0..=0x1f8010bf => todo!("write at dma3 (cdrom)"),
            0x1f8010c0..=0x1f8010cf => todo!("write at dma4 (spu)"),
            0x1f8010d0..=0x1f8010df => todo!("write at dma5 (pio)"),
            0x1f8010e0..=0x1f8010ef => todo!("write at dma6 (otc)"),
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
#[derive(Debug)]
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
