use core::alloc::Allocator;

use heapless::LenType;
use heapless::deque::DequeInner;
use heapless::vec::{VecInner, VecStorage};

use crate::{Allocate, Endecode, EndecodeError, Stream};

impl<A, T, LenT, S> Endecode<A> for VecInner<T, LenT, S>
where
    A: Allocator + Clone,
    T: Endecode<A> + Allocate<A> + Clone,
    LenT: LenType,
    S: VecStorage<T> + ?Sized,
{
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), crate::EndecodeError> {
        let mut len = self.len() as u64;
        sink.u64(&mut len)?;
        self.resize(len as usize, T::new_in(alloc.clone()))
            .map_err(|_| EndecodeError::Invalid)?;

        for el in self.iter_mut() {
            el.endecode(sink, alloc.clone())?;
        }

        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(None)
    }
}

impl<A, T, S> Endecode<A> for DequeInner<T, S>
where
    A: Allocator + Clone,
    S: VecStorage<T>,
    T: Endecode<A> + Allocate<A>,
{
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        let mut len = self.storage_len() as u64;
        sink.u64(&mut len)?;

        while len < self.storage_len() as u64 {
            self.pop_back();
        }
        while len > self.storage_len() as u64 {
            self.push_back(T::new_in(alloc.clone()))
                .map_err(|_| EndecodeError::Invalid)?;
        }

        for el in self.iter_mut() {
            el.endecode(sink, alloc.clone())?;
        }

        Ok(())
    }

    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(None)
    }
}
