use core::alloc::Allocator;
use core::cmp;

use heapless::BinaryHeap;
use heapless::binary_heap::Min;
use pchan_encode::Allocate;
use pchan_encode::derive::Endecode;

use crate::Emu;

/// for compatiblity with older code
pub type EventId = u64;

#[derive(Debug, Clone, Copy, Endecode, Default)]
pub enum EventKind {
    #[default]
    Nop,
    // spu
    SpuClock,
    // sio
    Sio0ProcTx,
    Sio0Irq,
    Sio0Ack,
    Sio0Transfer,
    // dma
    DmaTransfer,
    // cdrom
    CdromResponse(u32),
    CdromIrq,
}

#[derive(Debug, Clone, Copy, Endecode, Default)]
#[endecode(alloc(default))]
pub enum Task {
    #[default]
    #[endecode(alloc(default))]
    Nop,

    GpuFlushDrawCalls,
    GpuStartBlitVramToCpu,
    GpuIssueDrawCall,

    DmaStartTransfer,
}

/// `pchan` event queue
#[derive(derive_more::Debug, Clone, Endecode)]
pub struct Evque<A: Allocator> {
    clock: u64,
    #[debug(skip)]
    queue: Box<BinaryHeap<PchanEvent, Min, 512>, A>,
    #[debug(skip)]
    tasks: Vec<Task, A>,
}

#[derive(Default)]
pub struct EvCtx {
    pub id:        u64,
    pub clock:     u64,
    pub overshoot: u64,
}

#[derive(Debug, Clone, Endecode)]
#[endecode(alloc(default))]
pub struct PchanEvent {
    at_cycle: u64,
    event:    EventKind,
    id:       u64,
}

impl<A: Allocator + Copy> Allocate<A> for Evque<A> {
    fn new_in(alloc: A) -> Self {
        Evque {
            clock: 0,
            queue: Box::new_in(BinaryHeap::new(), alloc),
            tasks: Vec::with_capacity_in(2048, alloc),
        }
    }
}

impl Ord for PchanEvent {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.at_cycle.cmp(&other.at_cycle)
    }
}

impl PartialOrd for PchanEvent {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PchanEvent {
    fn eq(&self, other: &Self) -> bool {
        self.at_cycle == other.at_cycle
    }
}

impl Eq for PchanEvent {}

impl<A: Allocator + Copy> Emu<A> {
    pub async fn evque_advance(&mut self, d_clock: u64) {
        self.evque.clock = self.evque.clock.wrapping_add(d_clock);
        let clock = self.cpu.cycles;
        while let Some(ev) = self.evque.pop_next() {
            let ctx = EvCtx {
                id: ev.id,
                clock,
                overshoot: clock.saturating_sub(ev.at_cycle),
            };
            match ev.event {
                EventKind::Nop => {}
                EventKind::Sio0ProcTx => {
                    self.handle_ev_sio0_tx_proc(ctx);
                }
                EventKind::Sio0Irq => {
                    self.handle_ev_sio0_irq(ctx);
                }
                EventKind::Sio0Ack => {
                    self.handle_ev_sio0_ack(ctx);
                }
                EventKind::Sio0Transfer => {
                    self.handle_ev_sio0_transfer(ctx);
                }
                EventKind::SpuClock => {
                    self.handle_ev_spu_clock(ctx);
                }
                EventKind::DmaTransfer => {
                    self.handle_ev_dma_transfer(ctx).await;
                }
                EventKind::CdromResponse(id) => {
                    self.handle_ev_cdrom_response(ctx, id);
                }
                EventKind::CdromIrq => {
                    self.handle_ev_cdrom_irq(ctx);
                }
            };
        }
        let mut i = 0;
        while i < self.evque.tasks.len() {
            let task = self.evque.tasks[i];
            i += 1;
            self.run_task(task).await;
        }
        self.evque.tasks.clear();
    }

    async fn run_task(&mut self, task: Task) {
        match task {
            Task::Nop => {}
            Task::GpuFlushDrawCalls => {
                self.gpu.task_gpu_flush_draw_calls().await;
            }
            Task::GpuStartBlitVramToCpu => {
                self.task_gpu_start_blit_vram_to_cpu().await;
            }
            Task::GpuIssueDrawCall => {
                self.task_gpu_issue_draw_call().await;
            }
            Task::DmaStartTransfer => {
                self.task_dma_start_transfer().await;
            }
        }
    }
}

impl<A: Allocator> Evque<A> {
    fn pop_next(&mut self) -> Option<PchanEvent> {
        match self.queue.peek() {
            Some(ev) if ev.at_cycle <= self.clock => self.queue.pop(),
            _ => None,
        }
    }
    pub fn schedule_from(&mut self, kind: EventKind, id: u64, from_clock: u64, in_cycles: u64) {
        let ev = PchanEvent {
            at_cycle: from_clock.wrapping_add(in_cycles),
            event: kind,
            id,
        };
        let res = self.queue.push(ev);
        debug_assert!(res.is_ok(), "evque is too small");
    }

    pub fn schedule(&mut self, kind: EventKind, id: u64, in_cycles: u64) {
        self.schedule_from(kind, id, self.clock, in_cycles);
    }

    pub fn spawn(&mut self, task: Task) {
        self.tasks
            .push_within_capacity(task)
            .expect("task buf reallocation");
    }
}
