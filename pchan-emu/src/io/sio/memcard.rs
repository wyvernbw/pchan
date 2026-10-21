use pchan_encode::derive::Endecode;
use pchan_utils::hex;

use crate::io::sio::{Peripheral, SioCtrlReg};

#[derive(Default, derive_more::Debug, Clone, Endecode)]
pub struct Memcard {
    pub connected: bool,
}

impl Peripheral for Memcard {
    fn on_tx_write(
        &mut self,
        byte: u8,
        rx: &mut super::Sio0Rx,
        ctrl: &mut SioCtrlReg,
        irq: &mut bool,
    ) -> super::TxWriteResult {
        tracing::info!("memcard recv: {}", hex(byte));
        rx.send(0xff, ctrl, irq);
        super::TxWriteResult::TransferFinished
    }

    fn deselect(&mut self) {}
}
