use core::alloc::Allocator;

use smallvec::{Array, SmallVec};

use crate::{Allocate, Endecode};

impl<A: Allocator + Clone, S: Array> Endecode<A> for SmallVec<S>
where
    S::Item: Allocate<A> + Endecode<A>,
{
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        alloc: A,
    ) -> Result<(), crate::EndecodeError> {
        let mut len = self.len() as u64;
        sink.u64(&mut len)?;
        self.resize_with(len as usize, || S::Item::new_in(alloc.clone()));

        for el in self.iter_mut() {
            el.endecode(sink, alloc.clone())?;
        }

        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(None)
    }
}
