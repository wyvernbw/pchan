use core::alloc::Allocator;

use alloc::alloc::Global;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{Allocate, Encoder, EncoderSink, Endecode, EndecodeError, IntoEncoderSink, Stream};

impl<T, A: Allocator> Allocate<A> for Vec<T, A> {
    fn new_in(alloc: A) -> Self {
        Vec::new_in(alloc)
    }
}

impl<T: Endecode<A> + Allocate<A>, A: Allocator + Clone> Endecode<A> for Vec<T, A> {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        let mut len = self.len() as u64;
        sink.u64(&mut len)?;

        if len != self.len() as u64 {
            sink.check_remaining::<Vec<T, A>>(len as usize)?;

            let alloc = self.allocator().clone();
            self.resize_with(len as usize, || T::new_in(alloc.clone()));
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

impl<T: Allocate<A>, A: Allocator + Clone> Allocate<A> for Box<T, A> {
    fn new_in(alloc: A) -> Self {
        Box::new_in(T::new_in(alloc.clone()), alloc)
    }
}

impl<T: Endecode<A> + Allocate<A>, A: Allocator + Clone> Endecode<A> for Box<T, A> {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        T::endecode(self, sink, alloc)
    }

    fn size() -> Result<Option<usize>, EndecodeError> {
        T::size()
    }
}

impl<T: Endecode<A> + Allocate<A>, A: Allocator + Clone> Endecode<A> for Box<[T], A> {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        let mut len = self.len() as u64;
        sink.u64(&mut len)?;

        if len != self.len() as u64 {
            let alloc = Box::allocator(self);
            let mut temp = Vec::new_in(alloc.clone());
            temp.resize_with(len as usize, || T::new_in(alloc.clone()));
            *self = temp.into_boxed_slice();
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

impl<A: Allocator> Endecode<A> for String {
    fn endecode(&mut self, sink: &mut impl Stream, _: A) -> Result<(), EndecodeError> {
        struct StringFix<'a>(&'a mut Vec<u8>);
        impl Drop for StringFix<'_> {
            fn drop(&mut self) {
                self.0.clear();
            }
        }

        let mut len = self.len() as u64;
        sink.u64(&mut len)?;
        let len = len as usize;

        let buf = unsafe { self.as_mut_vec() };
        let buf = StringFix(buf);
        buf.0.truncate(len);

        if len > buf.0.len() {
            buf.0.resize(len, 0);
        }

        sink.bytes_no_order(buf.0)?;
        core::str::from_utf8(buf.0)?;

        core::mem::forget(buf);

        Ok(())
    }

    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(None)
    }
}

impl Allocate<Global> for String {
    fn new_in(_: Global) -> Self {
        String::new()
    }
}

pub struct GrowableSink<A: Allocator> {
    inner:  Vec<u8, A>,
    cursor: usize,
}

impl<A: Allocator> GrowableSink<A> {
    fn grow_if_needed(&mut self, more: usize) {
        if self.rem_len() < more {
            self.inner.reserve(more);
            self.inner.resize(self.inner.capacity(), 0);
        }
    }

    pub fn into_bytes(mut self) -> Vec<u8, A> {
        self.inner.truncate(self.cursor);
        self.inner
    }

    pub fn bytes(&self) -> &[u8] {
        &self.inner[..self.cursor]
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.inner[..self.cursor]
    }
}

impl<A: Allocator> Encoder<GrowableSink<A>> {
    pub fn bytes(&self) -> &[u8] {
        self.sink.bytes()
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.sink.bytes_mut()
    }
}

impl<A: Allocator> EncoderSink for GrowableSink<A> {
    fn take<const N: usize, T>(&mut self) -> Result<&mut [u8; N], EndecodeError> {
        self.grow_if_needed(N);

        let cursor = self.cursor;
        let head = self
            .inner
            .get_mut(cursor..cursor + N)
            .ok_or(EndecodeError::overflow::<T>())?;

        self.cursor += N;
        head.try_into().map_err(|_| EndecodeError::overflow::<T>())
    }

    fn take_dyn<T>(&mut self, len: usize) -> Result<&mut [u8], EndecodeError> {
        self.grow_if_needed(len);

        let cursor = self.cursor;
        self.cursor += len;

        let head = self
            .inner
            .get_mut(cursor..cursor + len)
            .ok_or(EndecodeError::overflow::<T>())?;
        Ok(head)
    }

    fn rem_len(&self) -> usize {
        self.inner.len().saturating_sub(self.cursor)
    }
}

impl<A: Allocator> IntoEncoderSink for Vec<u8, A> {
    type Sink = GrowableSink<A>;

    fn into_encoder_sink(self) -> Self::Sink {
        GrowableSink {
            inner:  self,
            cursor: 0,
        }
    }
}
