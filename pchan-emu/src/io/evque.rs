use core::alloc::Allocator;
use core::cmp;

use heapless::BinaryHeap;
use heapless::binary_heap::Min;
use pchan_encode::{Allocate, Endecode};
use slotmap::{KeyData, new_key_type};

use crate::Emu;

/// `pchan` event queue
#[derive(derive_more::Debug, Clone)]
pub struct Evque<T: ?Sized, A: Allocator> {
    clock: u64,
    #[debug(skip)]
    queue: Box<BinaryHeap<PchanEvent<T>, Min, 512>, A>,
}

new_key_type! {
    pub struct EventId;
}

impl<A: Allocator> Endecode<A> for EventId {
    fn endecode(
        &mut self,
        sink: &mut impl pchan_encode::Stream,
        alloc: A,
    ) -> Result<(), pchan_encode::EndecodeError> {
        self.0.endecode(sink, alloc)
    }

    fn size() -> Result<Option<usize>, pchan_encode::EndecodeError> {
        <KeyData as Endecode<A>>::size()
    }
}

impl<A: Allocator> Allocate<A> for EventId {
    fn new_in(_: A) -> Self {
        Self::default()
    }
}

#[derive(Default)]
pub struct EvCtx {
    pub id:        EventId,
    pub clock:     u64,
    pub overshoot: u64,
}

pub type PchanEventFn<T> = fn(&mut T, ctx: EvCtx);

#[derive(Debug, Clone)]
pub struct PchanEvent<T: ?Sized> {
    at_cycle: u64,
    fnptr:    PchanEventFn<T>,
    id:       EventId,
}

impl<A: Allocator, T> Allocate<A> for Evque<T, A> {
    fn new_in(alloc: A) -> Self {
        Evque {
            clock: 0,
            queue: Box::new_in(BinaryHeap::new(), alloc),
        }
    }
}

impl<T: ?Sized> Ord for PchanEvent<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.at_cycle.cmp(&other.at_cycle)
    }
}

impl<T: ?Sized> PartialOrd for PchanEvent<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: ?Sized> PartialEq for PchanEvent<T> {
    fn eq(&self, other: &Self) -> bool {
        self.at_cycle == other.at_cycle
    }
}

impl<T: ?Sized> Eq for PchanEvent<T> {}

impl<A: Allocator> Emu<A> {
    pub fn evque_advance(&mut self, d_clock: u64) {
        self.evque.clock = self.evque.clock.wrapping_add(d_clock);
        let clock = self.cpu.cycles;
        while let Some(ev) = self.evque.pop_next() {
            (ev.fnptr)(
                self,
                EvCtx {
                    id: ev.id,
                    clock,
                    overshoot: clock.saturating_sub(ev.at_cycle),
                },
            );
        }
    }
}

impl<T: ?Sized, A: Allocator> Evque<T, A> {
    fn pop_next(&mut self) -> Option<PchanEvent<T>> {
        match self.queue.peek() {
            Some(ev) if ev.at_cycle <= self.clock => self.queue.pop(),
            _ => None,
        }
    }
    pub fn schedule_from(
        &mut self,
        cb: PchanEventFn<T>,
        id: EventId,
        from_clock: u64,
        in_cycles: u64,
    ) {
        let ev = PchanEvent {
            at_cycle: from_clock.wrapping_add(in_cycles),
            fnptr: cb,
            id,
        };
        let res = self.queue.push(ev);
        debug_assert!(res.is_ok(), "evque is too small");
    }

    pub fn schedule(&mut self, cb: PchanEventFn<T>, id: EventId, in_cycles: u64) {
        self.schedule_from(cb, id, self.clock, in_cycles);
    }
}
