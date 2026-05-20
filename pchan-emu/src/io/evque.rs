use heapless::{BinaryHeap, binary_heap::Min};

use crate::Bus;

/// `pchan` event queue
#[derive(Debug, Clone)]
pub struct Evque<T: ?Sized> {
    clock: u64,
    queue: BinaryHeap<PchanEvent<T>, Min, 128>,
}

pub type PchanEventFn<T> = fn(&mut T, id: usize, clock: u64);

#[derive(Debug, Clone)]
pub struct PchanEvent<T: ?Sized> {
    at_cycle: u64,
    fnptr:    PchanEventFn<T>,
    id:       usize,
}

impl<T> Default for Evque<T> {
    fn default() -> Self {
        Self {
            queue: BinaryHeap::new(),
            clock: 0,
        }
    }
}

impl<T: ?Sized> Ord for PchanEvent<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.at_cycle.cmp(&other.at_cycle)
    }
}

impl<T: ?Sized> PartialOrd for PchanEvent<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: ?Sized> PartialEq for PchanEvent<T> {
    fn eq(&self, other: &Self) -> bool {
        self.at_cycle == other.at_cycle
    }
}

impl<T: ?Sized> Eq for PchanEvent<T> {}

pub trait EventQueue: Bus {
    fn evque_advance(&mut self, d_clock: u64) {
        self.evque_mut().clock = self.evque_mut().clock.wrapping_add(d_clock);
        while let Some(ev) = self.evque_mut().pop_next() {
            (ev.fnptr)(self, ev.id, ev.at_cycle);
        }
    }
}

impl<T> EventQueue for T where T: Bus {}

impl<T: ?Sized> Evque<T> {
    fn pop_next(&mut self) -> Option<PchanEvent<T>> {
        match self.queue.peek() {
            Some(ev) if ev.at_cycle <= self.clock => self.queue.pop(),
            _ => None,
        }
    }

    pub fn schedule(&mut self, cb: PchanEventFn<T>, id: usize, in_cycles: u64) {
        let ev = PchanEvent {
            at_cycle: self.clock.wrapping_add(in_cycles),
            fnptr: cb,
            id,
        };
        let res = self.queue.push(ev);
        debug_assert!(res.is_ok(), "evque is too small");
    }
}
