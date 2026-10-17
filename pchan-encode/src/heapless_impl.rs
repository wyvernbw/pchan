use core::alloc::Allocator;
use heapless::deque::DequeInner;
use heapless::vec::{VecInner, VecStorage};
use heapless::{BinaryHeap, LenType};

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

impl<A, T, K, const N: usize> Endecode<A> for BinaryHeap<T, K, N>
where
    A: Allocator + Clone,
    T: Endecode<A> + Allocate<A> + Clone + Ord,
    K: heapless::binary_heap::Kind,
{
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        let mut iter = self.iter_mut();
        let mut vec = iter.as_mut_slice();
        vec.endecode(sink, alloc)
    }

    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(None)
    }
}

impl<T, K, A: Allocator, const N: usize> Allocate<A> for BinaryHeap<T, K, N> {
    fn new_in(_: A) -> Self {
        Self::new()
    }
}
