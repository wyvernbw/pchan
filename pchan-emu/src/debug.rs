use derive_more as d;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct DebuggerState {
    pub breakpoints: HashMap<u32, Breakpoint>,
    pub stopped_on:  Option<Breakpoint>,
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

impl DebuggerState {
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
