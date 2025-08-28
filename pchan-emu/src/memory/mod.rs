pub const fn kb(value: usize) -> usize {
    value * 1024
}

pub fn buffer(size: usize) -> Box<[u8]> {
    vec![0u8; size].into_boxed_slice()
}

const MEM_SIZE: usize = kb(2048) + kb(8192) + kb(1) + kb(8) + kb(8) + kb(2048) + kb(512) + 512;

pub struct Memory(Box<[u8]>);

impl Default for Memory {
    fn default() -> Self {
        Memory(buffer(MEM_SIZE))
    }
}

pub struct PhysAddr(u32);

impl PhysAddr {
    fn try_new(address: u32) -> Option<Self> {
        let header = address & 0xE000_0000;
        match header {
            // KSEG0
            0x8000_0000 => Some(PhysAddr(address - 0x8000_0000)),
            // KSEG1
            0xA000_0000 => Some(PhysAddr(address - 0xA000_0000)),
            _ => None,
        }
    }
    fn new(address: u32) -> Self {
        PhysAddr::try_new(address).expect(&format!("unmapped address: {address}"))
    }
}

#[cfg(test)]
mod physaddr_tests {
    use super::PhysAddr;

    #[test]
    fn kseg0_to_physical() {
        let addr = PhysAddr::new(0x8000_0000);
        assert_eq!(addr.0, 0x0000_0000);

        let addr = PhysAddr::new(0x8020_0100);
        assert_eq!(addr.0, 0x0020_0100);
    }

    #[test]
    fn kseg1_to_physical() {
        let addr = PhysAddr::new(0xA000_0000);
        assert_eq!(addr.0, 0x0000_0000);

        let addr = PhysAddr::new(0xA020_0100);
        assert_eq!(addr.0, 0x0020_0100);
    }

    #[test]
    #[should_panic(expected = "unmapped address")]
    fn unmapped_address_panics() {
        PhysAddr::new(0x0000_0000); // KUSEG (unmapped)
    }

    #[test]
    fn try_new_returns_none_for_unmapped() {
        assert!(PhysAddr::try_new(0x0000_0000).is_none());
        assert!(PhysAddr::try_new(0xC000_0000).is_none()); // KSEG2
    }
}
