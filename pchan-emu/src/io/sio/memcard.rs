use pchan_encode::derive::Endecode;
use pchan_utils::hex;

use crate::io::sio::Peripheral;

#[derive(Default, derive_more::Debug, Clone, Endecode)]
pub struct Memcard {
    pub connected: bool,
}

impl Peripheral for Memcard {
    fn on_tx_write(&mut self, byte: u8, rx: &mut super::Sio0Rx) -> super::TxWriteResult {
        tracing::info!("memcard recv: {}", hex(byte));
        rx.push(0xff);
        super::TxWriteResult::TransferFinished
    }
}
