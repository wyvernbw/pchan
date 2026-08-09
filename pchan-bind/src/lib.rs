pub mod input;

use std::fmt::Debug;
use std::sync::Arc;

use ringbuf::storage::Heap;
use ringbuf::traits::Split;
use ringbuf::{CachingCons, CachingProd, SharedRb};

pub use ringbuf;

pub struct AudioConsumer {
    pub cons: CachingCons<Arc<SharedRb<Heap<i16>>>>,
}

impl Debug for AudioConsumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioConsumer").finish()
    }
}

pub struct AudioProducer {
    pub prod: CachingProd<Arc<SharedRb<Heap<i16>>>>,
}

impl Debug for AudioProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioProducer").finish()
    }
}

pub fn create_audio() -> (AudioConsumer, AudioProducer) {
    let rb = SharedRb::<Heap<i16>>::new(4096 * 8); // 32kb audio ringbuffer
    let (prod, cons) = rb.split();
    (AudioConsumer { cons }, AudioProducer { prod })
}

pub trait BindAudioConsumer {
    fn bind_consumer(&mut self, cons: AudioConsumer);
}

pub trait BindAudioProducer {
    fn bind_producer(&mut self, prod: AudioProducer);
}

pub fn bind_audio(bcons: &mut impl BindAudioConsumer, bprod: &mut impl BindAudioProducer) {
    let (cons, prod) = create_audio();
    bcons.bind_consumer(cons);
    bprod.bind_producer(prod);
}
