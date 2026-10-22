use core::alloc::Allocator;

pub use glam::*;

use crate::{Allocate, Endecode};

impl<A: Allocator> Endecode<A> for U8Vec2 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.u8(&mut self.x)?;
        sink.u8(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for U8Vec3 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.u8(&mut self.x)?;
        sink.u8(&mut self.y)?;
        sink.u8(&mut self.z)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for U8Vec4 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.u8(&mut self.x)?;
        sink.u8(&mut self.y)?;
        sink.u8(&mut self.z)?;
        sink.u8(&mut self.w)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for U16Vec2 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.u16(&mut self.x)?;
        sink.u16(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for U64Vec2 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.u64(&mut self.x)?;
        sink.u64(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for I16Vec2 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.i16(&mut self.x)?;
        sink.i16(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for I16Vec3 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.i16(&mut self.x)?;
        sink.i16(&mut self.y)?;
        sink.i16(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for IVec2 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.i32(&mut self.x)?;
        sink.i32(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

impl<A: Allocator> Endecode<A> for IVec3 {
    fn endecode(
        &mut self,
        sink: &mut impl crate::Stream,
        _: A,
    ) -> Result<(), crate::EndecodeError> {
        sink.i32(&mut self.x)?;
        sink.i32(&mut self.y)?;
        sink.i32(&mut self.y)?;
        Ok(())
    }

    fn size() -> Result<Option<usize>, crate::EndecodeError> {
        Ok(Some(size_of::<Self>()))
    }
}

macro_rules! impl_allocate_default {
    ($ty:ty) => {
        impl<A: Allocator> Allocate<A> for $ty {
            fn new_in(_: A) -> Self {
                Self::default()
            }
        }
    };
}

impl_allocate_default!(U8Vec2);
impl_allocate_default!(U8Vec3);
impl_allocate_default!(U16Vec2);
impl_allocate_default!(U16Vec3);
impl_allocate_default!(UVec2);
impl_allocate_default!(UVec3);
impl_allocate_default!(U64Vec2);
impl_allocate_default!(U64Vec3);

impl_allocate_default!(I8Vec2);
impl_allocate_default!(I8Vec3);
impl_allocate_default!(I16Vec2);
impl_allocate_default!(I16Vec3);
impl_allocate_default!(IVec2);
impl_allocate_default!(IVec3);
impl_allocate_default!(I64Vec2);
impl_allocate_default!(I64Vec3);
