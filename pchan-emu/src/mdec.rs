use core::{fmt, mem};

use arbitrary_int::prelude::*;
use bitbybit::*;

use crate::io::{CastIOFrom, IOResult, UnhandledIO};
use crate::{Emu, trace_todo};

impl Emu {
    pub(crate) fn mdec_read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fff_ffff;
        match address {
            0x1f801820 => trace_todo!(0xff, "read at mdec response register"),
            0x1f801824 => Ok(self.mdec.stat.raw_value().io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    pub(crate) fn mdec_write(&mut self, address: u32, value: u32) -> IOResult<()> {
        let address = address & 0x1fff_ffff;
        match address {
            0x1f801820 => {
                self.mdec_cmd_send(MDecCmd::new_with_raw_value(value));
                Ok(())
            }
            0x1f801824 => {
                self.mdec_write_ctrl(MDecCtrl::new_with_raw_value(value));
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }
}

impl Emu {
    fn mdec_write_ctrl(&mut self, value: MDecCtrl) {
        if value.reset() {
            // TODO: abort mdec cmd
            self.mdec.stat = MDecStatus::new_with_raw_value(0x80040000);
        }
        let dma0on = value.data_in_toggle();
        let dma1on = value.data_out_toggle();
        self.dma.dpcr.set_dma0on(dma0on);
        self.dma.dpcr.set_dma1on(dma1on);
        self.mdec.stat.set_dreq0(dma0on);
        self.mdec.stat.set_dreq1(dma1on);
    }

    pub(crate) fn mdec_cmd_send(&mut self, cmd: MDecCmd) {
        match &mut self.mdec.cmd {
            MDecCmdState::Idle => match cmd.cmd().value() {
                2 => {
                    self.mdec.stat_cpy_bits(cmd);
                    let cmd = cmd.to_cmd2();
                    self.mdec
                        .cmd_begin(MDecCmdState::Cmd2SQT(MDecCmd2SQTState { cmd, count: 0 }));
                }
                3 => {
                    self.mdec.stat_cpy_bits(cmd);
                    self.mdec
                        .cmd_begin(MDecCmdState::Cmd3SS(MDecCmd3SetScaleState { count: 0 }));
                }
                0..8 => {
                    todo!("mdec cmd{}", cmd.cmd())
                }
                8.. => unreachable!(),
            },
            MDecCmdState::Cmd2SQT(state) => {
                let params = cmd.as_bytes();
                for param in params {
                    tracing::trace!(target:"mdec", "sqt {:?} {}", state.cmd.color(), state.count);
                    match (state.cmd.color(), state.count) {
                        (MDecCmd2SQTColor::Luma | MDecCmd2SQTColor::LumaAndColor, 0..63) => {
                            self.mdec.luma_table[usize::from(state.count)] = param;
                            state.count += 1;
                        }
                        (MDecCmd2SQTColor::LumaAndColor, 63..127) => {
                            self.mdec.color_table[usize::from(state.count - 63)] = param;
                            state.count += 1;
                        }
                        (MDecCmd2SQTColor::Luma, 63..)
                        | (MDecCmd2SQTColor::LumaAndColor, 127..) => {
                            self.mdec.cmd_end();
                            break;
                        }
                    }
                }
            }
            MDecCmdState::Cmd3SS(state) => {
                let params = unsafe { mem::transmute::<_, [i16; 2]>(cmd.raw_value()) };
                match state.count {
                    0..63 => {
                        self.mdec.scale_table[usize::from(state.count)] = params[0];
                        self.mdec.scale_table[usize::from(state.count + 1)] = params[1];
                        state.count += 2;
                    }
                    63.. => {
                        self.mdec.cmd_end();
                    }
                }
            }
        }
    }
}

impl MDec {
    fn cmd_begin(&mut self, cmd: MDecCmdState) {
        tracing::info!(target: "mdec", "{:<14.14}", cmd);
        self.stat.set_busy(true);
        self.cmd = cmd;
    }
    fn cmd_end(&mut self) {
        self.stat.set_busy(false);
        self.cmd = MDecCmdState::Idle;
    }
    fn stat_cpy_bits(&mut self, cmd: MDecCmd) {
        self.stat.set_cpy_dst(cmd.cpy_src());
    }
}

#[derive(Default, Debug, Clone)]
pub struct MDec {
    stat: MDecStatus,
    cmd:    MDecCmdState,

    luma_table:  [u8; 64] = [0; 64],
    color_table: [u8; 64] = [0; 64],
    scale_table: [i16; 64] = [0; 64]
}

/// ```plaintext
///  31    FIFO1  Data-Out Fifo Empty (0=No, 1=Empty)
///  30    FIFO0  Data-In Fifo Full   (0=No, 1=Full, or Last word received)
///  29    BUSY   Command Busy  (0=Ready, 1=Busy receiving or processing parameters)
///  28    DREQ0  Data-In Request  (set when DMA0 enabled and ready to receive data)
///  27    DREQ1  Data-Out Request (set when DMA1 enabled and ready to send data)
///  26-25 ?      Data Output Depth  (0=4bit, 1=8bit, 2=24bit, 3=15bit)      ;CMD.28-27
///  24    ?      Data Output Signed (0=Unsigned, 1=Signed)                  ;CMD.26
///  23    STP    Data Output Bit15  (0=Clear, 1=Set) (for 15bit depth only) ;CMD.25
///  22-19        Not used (seems to be always zero)
///  18-16 ?      Current Block (0..3=Y1..Y4, 4=Cr, 5=Cb) (or for mono: always 4=Y)
///  15-0  ?      Number of Parameter Words remaining minus 1  (FFFFh=None)  ;CMD.Bit0-15
/// ```
#[bitfield(u32)]
#[derive(Default, Debug)]
struct MDecStatus {
    #[bit(31, rw)]
    fifo1: bool,
    #[bit(30, rw)]
    fifo0: bool,
    #[bit(29, rw)]
    busy:  bool,
    #[bit(28, rw)]
    dreq0: bool,
    #[bit(27, rw)]
    dreq1: bool,

    #[bits(25..=26, rw)]
    out_depth: MDecOutDepth,
    #[bit(24, rw)]
    out_sign:  bool,
    #[bit(23)]
    out_stp:   bool,

    /// MDEC(2,3) cmds sets these bits
    #[bits(23..=26, rw)]
    cpy_dst: u4,

    #[bits(16..=18, rw)]
    curr_block: MDecBlock,
}

#[bitenum(u2, exhaustive = true)]
#[allow(clippy::enum_variant_names)]
#[derive(Default, Debug)]
enum MDecOutDepth {
    #[default]
    D4Bit  = 0,
    D8Bit  = 1,
    D24Bit = 2,
    D15Bit = 3,
}

#[bitenum(u3, exhaustive = true)]
#[derive(Default, Debug)]
enum MDecBlock {
    #[default]
    Y1   = 0,
    Y2   = 1,
    Y3   = 2,
    Y4   = 3,
    Cr   = 4,
    Cb   = 5,
    NA01 = 6,
    NA02 = 7,
}

/// ```plaintext
///  31    Reset MDEC (0=No change, 1=Abort any command, and set status=80040000h)
///  30    Enable Data-In Request  (0=Disable, 1=Enable DMA0 and Status.bit28)
///  29    Enable Data-Out Request (0=Disable, 1=Enable DMA1 and Status.bit27)
///  28-0  Unknown/Not used - usually zero
/// ```
#[bitfield(u32)]
#[must_use]
#[derive(Debug)]
pub struct MDecCtrl {
    #[bit(31, r)]
    reset:           bool,
    #[bit(30, r)]
    data_in_toggle:  bool,
    #[bit(29, r)]
    data_out_toggle: bool,
}

#[bitfield(u32)]
#[must_use]
pub struct MDecCmd {
    #[bits(29..=31, rw)]
    cmd:     u3,
    #[bits(25..=28, rw)]
    cpy_src: u4,
}

impl MDecCmd {
    fn as_bytes(self) -> [u8; 4] {
        self.raw_value().to_le_bytes()
    }
    fn to_cmd2(self) -> MDecCmd2SQT {
        MDecCmd2SQT::new_with_raw_value(self.raw_value())
    }
}

#[derive(Debug, Default, Clone)]
enum MDecCmdState {
    #[default]
    Idle,
    Cmd2SQT(MDecCmd2SQTState),
    Cmd3SS(MDecCmd3SetScaleState),
}

impl fmt::Display for MDecCmdState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MDecCmdState::Idle => write!(f, "Idle"),
            MDecCmdState::Cmd2SQT(_) => write!(f, "Cmd2SQT"),
            MDecCmdState::Cmd3SS(_) => write!(f, "Cmd3SS"),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct MDecCmd2SQTState {
    cmd:   MDecCmd2SQT,
    count: u8,
}

#[derive(Debug, Default, Clone)]
struct MDecCmd3SetScaleState {
    count: u8,
}

/// MDEC(1) - Decode Macroblock(s)
///
/// ```plaintext
/// 31-29 Command (1=decode_macroblock)
/// 28-27 Data Output Depth  (0=4bit, 1=8bit, 2=24bit, 3=15bit)      ;STAT.26-25
/// 26    Data Output Signed (0=Unsigned, 1=Signed)                  ;STAT.24
/// 25    Data Output Bit15  (0=Clear, 1=Set) (for 15bit depth only) ;STAT.23
/// 24-16 Not used (should be zero)
/// 15-0  Number of Parameter Words (size of compressed data)
/// ```
pub struct MDecCmd1Decode {}

/// MDEC(2) - Set Quant Table(s)
///
/// ```plaintext
/// 31-29 Command (2=set_iqtab)
/// 28-1  Not used (should be zero)  ;Bit25-28 are copied to STAT.23-26 though
/// 0     Color   (0=Luminance only, 1=Luminance and Color)
/// ```
#[bitfield(u32, debug)]
#[must_use]
#[derive(Default)]
pub struct MDecCmd2SQT {
    #[bits(25..=28, rw)]
    sqt_cpy_src: u4,
    #[bit(0, rw)]
    color:       MDecCmd2SQTColor,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum MDecCmd2SQTColor {
    Luma         = 0,
    LumaAndColor = 1,
}
