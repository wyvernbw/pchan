#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
mod collections;

use core::alloc::Allocator;
use core::any::type_name;
use core::marker::PhantomData;
use core::str::Utf8Error;

pub mod derive {
    pub use pchan_encode_derive::Endecode;
}

use thiserror::Error;

#[derive(Error, Clone, Debug)]
pub enum EndecodeError {
    #[error("sink buffer overflow when encoding {0}")]
    Overflow(&'static str),
    #[error("sink buffer underflow when decoding {0}")]
    Underflow(&'static str),
    #[error("invalid value for type")]
    Invalid,
    #[error(transparent)]
    Utf8Error(#[from] Utf8Error),
    #[error("invalid enum tag {0} for type {1}")]
    InvalidTag(u8, &'static str),
}

impl EndecodeError {
    fn overflow<T>() -> Self {
        Self::Overflow(type_name::<T>())
    }
    fn underflow<T>() -> Self {
        Self::Underflow(type_name::<T>())
    }
}

pub trait Endecode<A: Allocator> {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError>;
    fn size() -> Result<Option<usize>, EndecodeError>;
}

pub trait Allocate<A: Allocator> {
    fn new_in(alloc: A) -> Self;
}

macro_rules! num_allocate {
    ($ty:ty) => {
        impl<A: Allocator> Allocate<A> for $ty {
            fn new_in(_: A) -> Self {
                0
            }
        }
    };
}

num_allocate!(u8);
num_allocate!(u16);
num_allocate!(u32);
num_allocate!(u64);

pub trait EncoderSink {
    fn bytes(&self) -> &[u8];
    fn bytes_mut(&mut self) -> &mut [u8];
    fn take<const N: usize, T>(&mut self) -> Result<&mut [u8; N], EndecodeError>;
    fn take_dyn<T>(&mut self, len: usize) -> Result<&mut [u8], EndecodeError>;
    fn rem_len(&self) -> usize;
}

impl EncoderSink for &mut [u8] {
    fn bytes(&self) -> &[u8] {
        self
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self
    }

    fn take<const N: usize, T>(&mut self) -> Result<&mut [u8; N], EndecodeError> {
        let buf = core::mem::take(self);
        let (head, rest) = buf
            .split_first_chunk_mut::<N>()
            .ok_or(EndecodeError::overflow::<T>())?;
        *self = rest;
        Ok(head)
    }

    fn take_dyn<T>(&mut self, len: usize) -> Result<&mut [u8], EndecodeError> {
        let buf = core::mem::take(self);
        let (head, rest) = buf
            .split_at_mut_checked(len)
            .ok_or(EndecodeError::overflow::<T>())?;
        *self = rest;
        Ok(head)
    }

    fn rem_len(&self) -> usize {
        self.len()
    }
}

pub trait IntoEncoderSink {
    type Sink: EncoderSink;
    fn into_encoder_sink(self) -> Self::Sink;
}

impl<S> IntoEncoderSink for S
where
    S: EncoderSink,
{
    type Sink = Self;

    fn into_encoder_sink(self) -> Self::Sink {
        self
    }
}

pub struct Encoder<S> {
    sink: S,
}

pub struct Decoder<'a> {
    src: &'a [u8],
}

impl<'a> Decoder<'a> {
    fn take<const N: usize, T>(&mut self) -> Result<&'a [u8; N], EndecodeError> {
        let (head, rest) = self
            .src
            .split_first_chunk::<N>()
            .ok_or(EndecodeError::underflow::<T>())?;
        self.src = rest;
        Ok(head)
    }
    fn take_dyn<T>(&mut self, len: usize) -> Result<&'a [u8], EndecodeError> {
        let (head, rest) = self
            .src
            .split_at_checked(len)
            .ok_or(EndecodeError::underflow::<T>())?;
        self.src = rest;
        Ok(head)
    }
}

pub trait Stream {
    fn u8(&mut self, value: &mut u8) -> Result<(), EndecodeError>;
    fn u16(&mut self, value: &mut u16) -> Result<(), EndecodeError>;
    fn u32(&mut self, value: &mut u32) -> Result<(), EndecodeError>;
    fn u64(&mut self, value: &mut u64) -> Result<(), EndecodeError>;
    fn bytes_no_order(&mut self, value: &mut [u8]) -> Result<(), EndecodeError>;
    fn padding(&mut self, len: usize) -> Result<(), EndecodeError>;

    fn check_remaining<T>(&self, more: usize) -> Result<(), EndecodeError>;

    fn bool(&mut self, value: &mut bool) -> Result<(), EndecodeError> {
        let mut num = u8::from(*value);
        self.u8(&mut num)?;
        match num {
            0 => *value = false,
            1 => *value = true,
            _ => return Err(EndecodeError::Invalid),
        }
        Ok(())
    }
}

impl<S: EncoderSink> Stream for Encoder<S> {
    fn u8(&mut self, value: &mut u8) -> Result<(), EndecodeError> {
        self.sink
            .take::<1, u8>()?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn u16(&mut self, value: &mut u16) -> Result<(), EndecodeError> {
        self.sink
            .take::<2, u16>()?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn u32(&mut self, value: &mut u32) -> Result<(), EndecodeError> {
        self.sink
            .take::<4, u32>()?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn u64(&mut self, value: &mut u64) -> Result<(), EndecodeError> {
        self.sink
            .take::<8, u64>()?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn bytes_no_order(&mut self, value: &mut [u8]) -> Result<(), EndecodeError> {
        let head = self.sink.take_dyn::<&[u8]>(value.len())?;
        head.copy_from_slice(value);
        Ok(())
    }

    fn check_remaining<T>(&self, more: usize) -> Result<(), EndecodeError> {
        if more > self.sink.rem_len() {
            Err(EndecodeError::overflow::<T>())
        } else {
            Ok(())
        }
    }

    fn padding(&mut self, len: usize) -> Result<(), EndecodeError> {
        self.sink.take_dyn::<&[u8]>(len)?;
        Ok(())
    }
}

impl Stream for Decoder<'_> {
    fn u8(&mut self, value: &mut u8) -> Result<(), EndecodeError> {
        *value = self.take::<1, u8>()?[0];
        Ok(())
    }

    fn u16(&mut self, value: &mut u16) -> Result<(), EndecodeError> {
        *value = u16::from_le_bytes(*self.take::<2, u16>()?);
        Ok(())
    }

    fn u32(&mut self, value: &mut u32) -> Result<(), EndecodeError> {
        *value = u32::from_le_bytes(*self.take::<4, u32>()?);
        Ok(())
    }

    fn u64(&mut self, value: &mut u64) -> Result<(), EndecodeError> {
        *value = u64::from_le_bytes(*self.take::<8, u64>()?);
        Ok(())
    }

    fn bytes_no_order(&mut self, value: &mut [u8]) -> Result<(), EndecodeError> {
        let buf = core::mem::take(&mut self.src);
        let (head, rest) = buf
            .split_at_checked(value.len())
            .ok_or(EndecodeError::underflow::<&[u8]>())?;
        value.copy_from_slice(head);
        self.src = rest;
        Ok(())
    }

    fn check_remaining<T>(&self, more: usize) -> Result<(), EndecodeError> {
        if more > self.src.len() {
            Err(EndecodeError::overflow::<T>())
        } else {
            Ok(())
        }
    }

    fn padding(&mut self, len: usize) -> Result<(), EndecodeError> {
        self.take_dyn::<&[u8]>(len)?;
        Ok(())
    }
}

impl<A: Allocator> Endecode<A> for u8 {
    fn endecode(&mut self, sink: &mut impl Stream, _: A) -> Result<(), EndecodeError> {
        sink.u8(self)
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(1))
    }
}

impl<A: Allocator> Endecode<A> for u16 {
    fn endecode(&mut self, sink: &mut impl Stream, _: A) -> Result<(), EndecodeError> {
        sink.u16(self)
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(2))
    }
}

impl<A: Allocator> Endecode<A> for u32 {
    fn endecode(&mut self, sink: &mut impl Stream, _: A) -> Result<(), EndecodeError> {
        sink.u32(self)
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(4))
    }
}

impl<A: Allocator> Endecode<A> for u64 {
    fn endecode(&mut self, sink: &mut impl Stream, _: A) -> Result<(), EndecodeError> {
        sink.u64(self)
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(8))
    }
}

impl<A: Allocator> Endecode<A> for usize {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        (*self as u64).endecode(sink, alloc)
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(8))
    }
}

impl<const N: usize, T: Allocate<A>, A: Allocator + Clone> Allocate<A> for [T; N] {
    fn new_in(alloc: A) -> Self {
        core::array::from_fn(|_| T::new_in(alloc.clone()))
    }
}

impl<A: Allocator + Clone, T: Endecode<A>, const N: usize> Endecode<A> for [T; N] {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        for el in self.iter_mut() {
            el.endecode(sink, alloc.clone())?;
        }
        Ok(())
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        let size = T::size()?;
        match size {
            Some(size) => Ok(Some(size * N)),
            None => Ok(None),
        }
    }
}

impl<T, A: Allocator> Allocate<A> for Option<T> {
    fn new_in(_: A) -> Self {
        Self::default()
    }
}

impl<A: Allocator + Clone, T: Endecode<A> + Allocate<A>> Endecode<A> for Option<T> {
    fn endecode(&mut self, sink: &mut impl Stream, alloc: A) -> Result<(), EndecodeError> {
        let mut is_some = self.is_some();
        sink.bool(&mut is_some)?;
        if is_some {
            let mut value = T::new_in(alloc.clone());
            value.endecode(sink, alloc)?;
        } else {
            match T::size()? {
                Some(size) => {
                    sink.padding(size)?;
                }
                None => {
                    let mut value = T::new_in(alloc.clone());
                    value.endecode(sink, alloc)?;
                }
            }
        }
        Ok(())
    }
    fn size() -> Result<Option<usize>, EndecodeError> {
        let size = T::size()?;
        match size {
            Some(size) => Ok(Some(size + 1)),
            None => Ok(None),
        }
    }
}

impl<A: Allocator, T> Endecode<A> for PhantomData<T> {
    fn endecode(&mut self, _sink: &mut impl Stream, _alloc: A) -> Result<(), EndecodeError> {
        Ok(())
    }

    fn size() -> Result<Option<usize>, EndecodeError> {
        Ok(Some(0))
    }
}

pub fn encode<I: IntoEncoderSink, T: Endecode<A>, A: Allocator>(
    value: &mut T,
    sink: I,
    alloc: A,
) -> Result<I::Sink, EndecodeError> {
    let sink = sink.into_encoder_sink();
    let mut encoder = Encoder { sink };
    value.endecode(&mut encoder, alloc)?;
    Ok(encoder.sink)
}

pub fn decode_in_place<T: Endecode<A>, A: Allocator>(
    value: &mut T,
    bytes: impl AsRef<[u8]>,
    alloc: A,
) -> Result<(), EndecodeError> {
    value.endecode(
        &mut Decoder {
            src: bytes.as_ref(),
        },
        alloc,
    )
}
