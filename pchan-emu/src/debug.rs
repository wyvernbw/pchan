use core::alloc::Allocator;
use derive_more as d;
use pchan_utils::hex;
use std::collections::HashMap;
use std::hash::RandomState;

#[derive(derive_more::Debug, Clone)]
pub struct DebuggerState<A: Allocator> {
    pub breakpoints:     HashMap<u32, Breakpoint, RandomState, A>,
    pub stopped_on:      Option<Breakpoint>,
    pub kernel_fn_stack: Vec<KernelFrame, A>,
}

#[derive(derive_more::Debug, Clone)]
pub struct KernelFrame {
    #[debug("{}", hex(self.address))]
    pub address: u8,
    #[debug("{}", hex(self.fn_num))]
    pub fn_num:  u8,
    #[debug("{}", hex(self.ra))]
    pub ra:      u32,
    #[debug("{:?}", self.args.map(hex))]
    pub args:    [u32; 5],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Breakpoint {
    pub address: u32,
    pub kind:    BreakpointKind,
    pub enabled: bool,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    d::BitOr,
    d::BitOrAssign,
    d::BitAnd,
    d::BitAndAssign,
    d::Not,
)]
pub struct BreakpointKind(u8);

impl BreakpointKind {
    pub const NONE: BreakpointKind = BreakpointKind(0);
    pub const READ: BreakpointKind = BreakpointKind(1);
    pub const WRITE: BreakpointKind = BreakpointKind(1 << 1);
    pub const EXECUTE: BreakpointKind = BreakpointKind(1 << 2);

    #[must_use]
    pub fn contains(self, other: BreakpointKind) -> bool {
        self & other != Self::NONE
    }

    #[must_use]
    pub fn difference(self, other: BreakpointKind) -> Self {
        BreakpointKind(self.0 ^ other.0)
    }
}

impl<A: Allocator + Copy> DebuggerState<A> {
    pub fn new(alloc: A) -> Self {
        Self {
            breakpoints:     HashMap::new_in(alloc),
            stopped_on:      None,
            kernel_fn_stack: Vec::new_in(alloc),
        }
    }
    pub fn break_on(&mut self, addr: u32, kind: BreakpointKind) -> bool {
        if let Some(brk) = self.breakpoints.get(&(addr & 0x1fff_ffff)) {
            if !brk.enabled {
                return self.stopped_on.is_some();
            }

            if brk.kind.contains(kind) {
                self.stopped_on = Some(*brk);
            }
        }
        self.stopped_on.is_some()
    }

    pub fn remove_breakpoint(&mut self, addr: u32) {
        if self.stopped_on.as_ref() == self.breakpoints.get(&addr) {
            self.stopped_on = None;
        }
        self.breakpoints.remove(&addr);
    }
}

impl KernelFrame {
    #[must_use]
    pub fn new(address: u32, fn_num: u32, ra: u32, args: [u32; 5]) -> Self {
        Self {
            address: address as u8,
            fn_num: fn_num as u8,
            ra,
            args,
        }
    }
}
