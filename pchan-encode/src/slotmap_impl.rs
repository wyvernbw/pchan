use core::alloc::Allocator;

use slotmap::{Key, KeyData, SlotMap};

use crate::{Allocate, Endecode};

impl<A: Allocator> Endecode<A> for KeyData {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        let mut value = self.as_ffi();
        sink.u64(&mut value)?;
        *self = KeyData::from_ffi(value);
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(8))
    }
}

impl<A: Allocator> Allocate<A> for KeyData {
    fn new_in(_: A) -> Self {
        KeyData::default()
    }
}
