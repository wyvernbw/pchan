use std::collections::VecDeque;

use bitvec::prelude as bv;
use bv::Lsb0;
use derive_more as de;
use pchan_utils::array;

#[cfg(target_arch = "aarch64")]
type RegAllocBitMap = bv::BitArray<[u32; 1]>;

/// x86_only has 16 registers
#[cfg(target_arch = "x86_64")]
type RegAllocBitMap = bv::BitArray<[u16; 1]>;

#[derive(Debug, Clone)]
pub struct RegAlloc {
    pub loaded: bv::BitArray<[u32; 1]>,
    pub dirty: bv::BitArray<[u32; 1]>,
    pub allocated: RegAllocBitMap,
    pub allocatable: RegAllocBitMap,
    /// this is in guest register space
    pub priority: RegAllocBitMap,
    /// mapping from guest to host register.
    pub mapping: [Option<Reg>; 32],
    /// mapping from host to guest register. its consider valid for it to map to unallocated registers.
    pub reverse_mapping: [u8; 32],
    /// never traverse this - its slow as shit
    pub history: VecDeque<u8>,
}

impl Default for RegAlloc {
    fn default() -> Self {
        Self {
            loaded: Default::default(),
            dirty: Default::default(),
            allocated: Default::default(),

            #[cfg(target_arch = "aarch64")]
            allocatable: bv::bitarr![
                // we can safely use 22 registers on arm64
                // so about 70% capacity
                //
                // keep half of arg registers
                u32, Lsb0;
                0, 0, 0, 0, // x0-x3
                1, 1, 1, 1, // x4-x7
                1, // r8 is syscall register, might get trampled
                1, 1, 1, 1, 1, 1, 1, // r9-r15 temporary registers
                0, 0, 0, // r16-r18 platform registers, not ideal
                // r19-r28 callee saved, ideal
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0 // r29 and r30 - frame and link registers, r31 - zero register (or sp)
            ],

            #[cfg(target_arch = "aarch64")]
            priority: bv::bitarr![
                u32, Lsb0;
                0, // $zero doesnt need an allocation
                1, // $at is high prio
                1, 1, // $v0-$v1
                1, 1, 1, 1, // $a0-$a3
                1, 1, 1, 1, 1, 1, 1, 1, // $t0-$t7
                1, 1, 1, 1, 1, 1, 1, 1, // $s0-$s7
                0, 0, // extra temp registers $t8 and $t9
                0, 0, // kernel registers $k0 and $k1
                0, // $gp is rarely used
                1, // $sp
                0, 0, // $fp and $ra. $fp is rarely used, $ra is not typically used without a jump, which triggers writebacks anyways
            ],
            mapping: Default::default(),
            reverse_mapping: Default::default(),
            history: Default::default(),
        }
    }
}

/// register allocation eviction type
/// - `EvictToMemory` is basically a permanent eviction. caller is not responsible for restoring register.
/// - `EvictToStack` this is like temporarily assigning a register. the caller *is* responsible for restoring the register off the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegAllocError {
    EvictToMemory(Evicted<Guest>, Allocated),
    EvictToStack(Evicted<Guest>, Allocated),
    AlreadyAllocatedTo(Allocated),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, de::Deref, de::From)]
pub struct Evicted<T: RegisterType>(T);
#[derive(Debug, Clone, Copy, PartialEq, Eq, de::Deref, de::From)]
pub struct Allocated(Reg);

impl From<Allocated> for u8 {
    fn from(value: Allocated) -> Self {
        (*value).into()
    }
}

impl From<u8> for Allocated {
    fn from(value: u8) -> Self {
        Allocated(Reg::new(value))
    }
}

pub type Guest = u8;
pub type Host = Reg;

pub trait RegisterType {}
impl RegisterType for Guest {}
impl RegisterType for Host {}

pub type AllocResult = Result<Allocated, RegAllocError>;

impl RegAlloc {
    pub fn first_free(&self) -> Option<Reg> {
        let [allocated] = self.allocated.into_inner();
        let [allocatable] = self.allocatable.into_inner();
        let allocate_from = allocatable & (!allocated);
        let index = allocate_from.trailing_zeros();
        Self::ensure_reg(index as _).map(Reg::new)
    }

    pub fn is_full(&self) -> bool {
        self.first_free().is_none()
    }

    pub fn first_allocatable(&self) -> Reg {
        let [allocatable] = self.allocatable.into_inner();
        let index = allocatable.trailing_zeros() as u8;
        Reg::new(index)
    }

    pub fn allocated_guest_to_host(&self, guest_reg: u8) -> Option<Reg> {
        self.mapping[guest_reg as usize]
    }

    // # Returns
    // guest index of first allocated low priority.
    // use [`allocated_guest_to_host`] to map to host register.
    pub fn first_allocated_low_priority(&self) -> Option<u8> {
        self.mapping
            .iter()
            .enumerate()
            .position(|(guest_idx, host)| {
                host.is_some() && !self.priority[guest_idx] // LOW priority = !prio
            })
            .map(|idx| idx as _)
        // .and_then(|idx| Self::ensure_reg(idx as _))
    }

    fn update_history(&mut self, guest_reg: u8) {
        if self.history.len() >= 32 {
            self.history.pop_front();
        }
        self.history.push_back(guest_reg);
    }

    pub fn regalloc(&mut self, guest_reg: Guest) -> Result<Allocated, RegAllocError> {
        debug_assert!(guest_reg <= 31);

        if let Some(host) = self.allocated_guest_to_host(guest_reg) {
            return Err(RegAllocError::AlreadyAllocatedTo(host.into()));
        }

        let prio = self.priority[guest_reg as usize];
        let first_free = self.first_free();

        if let Some(free) = first_free {
            return Ok(self.alloc(guest_reg, free));
        }

        let first_low_prio = self.first_allocated_low_priority();

        match first_low_prio {
            Some(first_low_prio) => {
                let first_low_prio_host = self
                    .allocated_guest_to_host(first_low_prio)
                    .unwrap_or_else(|| unreachable!());

                let evicted_guest = self.evict_at(first_low_prio_host);
                let host_reg = self.alloc(guest_reg, first_low_prio_host);
                let guest_dirty = self.dirty[first_low_prio as usize];

                match (prio, guest_dirty) {
                    (_, true) => Err(RegAllocError::EvictToMemory(evicted_guest, host_reg)),
                    (true, false) => Err(RegAllocError::EvictToMemory(evicted_guest, host_reg)),
                    (false, false) => Err(RegAllocError::EvictToStack(evicted_guest, host_reg)),
                }
            }
            None => {
                let (evicted_guest, evicted_host) = self.evict_any();
                let guest_dirty = self.dirty[*evicted_guest as usize];
                let host = self.alloc(guest_reg, *evicted_host);
                match (prio, guest_dirty) {
                    (_, true) => Err(RegAllocError::EvictToMemory(evicted_guest, host)),
                    (true, false) => Err(RegAllocError::EvictToMemory(evicted_guest, host)),
                    (false, false) => Err(RegAllocError::EvictToStack(evicted_guest, host)),
                }
            }
        }
    }

    /// evict any register
    /// policy: oldest first
    fn evict_any(&mut self) -> (Evicted<Guest>, Evicted<Host>) {
        // debug_assert!(self.is_full(), "cannot evict any on non full reg allocator");
        // debug_assert!(
        //     self.first_allocated_low_priority().is_none(),
        //     "cannot evict any if there are low prio allocations. evict first low prio instead."
        // );

        loop {
            let Some(reg) = self.history.pop_front() else {
                let evicted_host = self
                    .first_allocated_low_priority()
                    .and_then(|reg| self.allocated_guest_to_host(reg))
                    .unwrap_or_else(|| self.first_allocatable());
                return (self.evict_at(evicted_host), Evicted(evicted_host));
            };
            let Some(reg) = self.allocated_guest_to_host(reg) else {
                continue;
            };
            return (self.evict_at(reg), Evicted(reg));
        }
    }

    fn evict_at(&mut self, host: Reg) -> Evicted<Guest> {
        let host = host.to_idx();
        self.allocated.set(host as usize, false);
        let guest = self.reverse_mapping[host as usize];
        self.mapping[guest as usize] = None;
        Evicted(guest)
    }

    fn alloc(&mut self, guest_reg: u8, host_reg: Reg) -> Allocated {
        let host_reg = host_reg.to_idx();
        self.allocated.set(host_reg as usize, true);
        self.update_history(guest_reg);

        #[cfg(target_arch = "aarch64")]
        {
            let reg = Reg::new(host_reg);
            self.mapping[guest_reg as usize] = Some(reg);
            self.reverse_mapping[host_reg as usize] = guest_reg;
            reg.into()
        }
    }

    fn ensure_reg(reg: u8) -> Option<u8> {
        match reg {
            #[cfg(target_arch = "aarch64")]
            32.. => None,

            other => Some(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reg {
    #[cfg(target_arch = "aarch64")]
    W(u8),
}

#[cfg(target_arch = "aarch64")]
impl Reg {
    const WZR: Self = Reg::W(31);
    const WSP: Self = Reg::W(31);

    const DELAY_1: Self = Reg::W(28);
    const DELAY_2: Self = Reg::W(27);

    pub const fn new(idx: u8) -> Self {
        Self::W(idx)
    }

    pub const fn to_idx(self) -> u8 {
        self.into()
    }

    pub fn consecutive(self: Reg, b: Reg) -> bool {
        let Reg::W(w) = self;
        b == Reg::W(w + 1)
    }

    pub fn caller_saved(&self) -> bool {
        matches!(self, Reg::W(19..=31))
    }
}

#[cfg(target_arch = "aarch64")]
impl const From<Reg> for u8 {
    fn from(value: Reg) -> Self {
        match value {
            Reg::W(reg) => reg,
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub static REG_MAP: [Option<Reg>; 32] = array![
    // $zero
    0 => Some(Reg::WZR),

    // x0-x3 are used for function arguments or as
    // temp registers.
    // x0 is always the pointer to the `Emu` struct.
    // r18 is the platform register and cannot be used.
    // x27-x28 should be kept for the delay slot.
    // $at
    1 => Some(Reg::W(4)),

    // $v0, $v1
    2 => Some(Reg::W(5)),
    3 => Some(Reg::W(6)),

    // $a0-$a3
    4 => Some(Reg::W(7)),
    5 => Some(Reg::W(8)),
    6 => Some(Reg::W(9)),
    7 => Some(Reg::W(10)),

    // $t0-$t7
    8 => Some(Reg::W(11)),
    9 => Some(Reg::W(12)),
    10 => Some(Reg::W(13)),
    11 => Some(Reg::W(14)),
    12 => Some(Reg::W(15)),
    13 => Some(Reg::W(16)),
    14 => Some(Reg::W(17)),
    // r18 must not be used.
    15 => Some(Reg::W(19)),

    // $s0-$s7
    16 => Some(Reg::W(20)),
    17 => Some(Reg::W(21)),
    18 => Some(Reg::W(22)),
    19 => Some(Reg::W(23)),
    20 => Some(Reg::W(24)),
    21 => Some(Reg::W(25)),
    22 => Some(Reg::W(26)),
    23 => Some(Reg::W(27)),

    // $t8-$t9
    24 => None,
    25 => None,

    // $k0-$k1
    26 => None,
    27 => None,

    // $gp
    28 => None,

    // $sp
    29 => Some(Reg::WSP),

    // $fp
    30 => None,

    // $ra
    31 => None
];

#[cfg(test)]
pub mod regalloc_tests {
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::*;

    #[test]
    pub fn regalloc_test_one_alloc() {
        let mut regalloc = RegAlloc::default();
        let result = regalloc.regalloc(16);
        assert!(result.is_ok());
        #[cfg(target_arch = "aarch64")]
        assert_eq!(result, Ok(4.into()));
    }

    #[rstest]
    #[case(1, [Ok(4.into())])]
    #[case(2, [Ok(4.into()), Ok(5.into())])]
    #[case(12, (4..16).map(Into::into).map(Ok).collect::<Vec<_>>())]
    #[case(22, (4..16).chain(19..=28).map(Into::into).map(Ok).collect::<Vec<_>>())]
    pub fn regalloc_test_multiple_alloc_free(
        #[case] count: usize,
        #[case] expected: impl Into<Vec<AllocResult>>,
    ) {
        let mut regalloc = RegAlloc::default();
        let result = (1..=count)
            .map(|reg| regalloc.regalloc(reg as u8))
            .collect::<Vec<_>>();
        assert_eq!(result, expected.into());
    }

    #[rstest]
    // 1 high priority register
    #[case([23], [Err(RegAllocError::EvictToMemory(Evicted(1), 4.into()))])]
    // 2 high priority registers
    #[case([23, 29], [Err(RegAllocError::EvictToMemory(Evicted(1), 4.into())), Err(RegAllocError::EvictToMemory(Evicted(2), 5.into()))])]
    // 1 low priority register
    #[case([24], [Err(RegAllocError::EvictToStack(Evicted(1), 4.into()))])]
    // 2 low priority register
    // this evicts a high priority register to the stack and allocates a low priority register in
    // its place. afterwards, that register gets evicted by another low priority register and in
    // the end they use the same host register 24
    #[case([24, 25], [Err(RegAllocError::EvictToStack(Evicted(1), 4.into())), Err(RegAllocError::EvictToStack(Evicted(24), 4.into()))])]
    pub fn regalloc_test_evictions_after_high_prio(
        #[case] registers_after_full: impl Into<Vec<u8>>,
        #[case] expected: impl Into<Vec<AllocResult>>,
    ) {
        let expected = expected.into();
        let registers_after_full = registers_after_full.into();
        let mut regalloc = RegAlloc::default();
        for i in 1..=22 {
            _ = regalloc.regalloc(i);
        }
        let results = registers_after_full
            .into_iter()
            .map(|reg| regalloc.regalloc(reg))
            .collect::<Vec<_>>();
        assert_eq!(results, expected)
    }

    #[rstest]
    pub fn regalloc_test_stresstest() {
        let mut regalloc = RegAlloc::default();
        for _ in 0..100000 {
            let reg: u8 = std::random::random(..);
            let reg = reg % 32;
            _ = regalloc.regalloc(reg);
        }
        assert!(
            regalloc.allocated.into_inner()[0].count_ones()
                <= regalloc.allocatable.into_inner()[0].count_ones()
        );
        assert!(regalloc.history.len() <= regalloc.allocated.into_inner()[0].count_ones() as usize);
        assert!(regalloc.is_full());
    }

    #[rstest]
    pub fn regalloc_test_alloc_same() {
        let mut regalloc = RegAlloc::default();
        let res_0 = regalloc.regalloc(16);
        let res_1 = regalloc.regalloc(16);
        assert_eq!(res_0, Ok(4.into()));
        assert_eq!(res_1, Err(RegAllocError::AlreadyAllocatedTo(4.into())));
    }
}
