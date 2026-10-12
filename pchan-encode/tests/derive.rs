use core::alloc::Allocator;
use core::marker::PhantomData;

use alloc::alloc::Global;
use derive_more as d;
use pchan_encode::derive::Endecode;
use pchan_encode::{EncoderSink, Endecode, decode_in_place, encode};

extern crate alloc;

#[derive(Endecode, Default, Debug, PartialEq, Eq, Clone)]
#[endecode(alloc(default))]
struct Basic {
    a: u32,
    b: u64,
    c: u8,
}

#[test]
fn test_basic() {
    let mut basic = Basic {
        a: 42,
        b: u64::MAX,
        c: 2,
    };
    let mut decoded = Basic::default();
    let sink = encode(&mut basic, Vec::with_capacity(10), Global).unwrap();
    decode_in_place(&mut decoded, sink.bytes(), Global).unwrap();

    assert_eq!(basic, decoded);
}

#[derive(Endecode, Debug)]
struct StructWithVec<A: Allocator> {
    a:     u8,
    items: Vec<u32, A>,
}

#[test]
fn test_struct_with_vec() {
    let mut basic = StructWithVec {
        a:     42,
        items: vec![8, 9, 42, 69],
    };
    let mut decoded = StructWithVec {
        a:     0,
        items: Vec::new(),
    };
    let sink = encode(&mut basic, Vec::with_capacity(10), Global).unwrap();
    decode_in_place(&mut decoded, sink.bytes(), Global).unwrap();

    assert_eq!(basic.a, decoded.a);
    assert_eq!(basic.items, decoded.items);
}

#[derive(Endecode, Debug)]
struct StructWithGenerics<A: Allocator, T: Endecode<A>> {
    a:      u32,
    inner:  T,
    _alloc: PhantomData<A>,
}

#[test]
fn test_struct_with_generics() {
    let mut basic = StructWithGenerics {
        a:      42,
        inner:  "hello world!".to_owned(),
        _alloc: PhantomData,
    };
    let mut decoded = StructWithGenerics {
        a:      0,
        inner:  String::new(),
        _alloc: PhantomData,
    };
    let sink = encode(&mut basic, Vec::with_capacity(10), Global).unwrap();
    decode_in_place(&mut decoded, sink.bytes(), Global).unwrap();

    assert_eq!(basic.a, decoded.a);
    assert_eq!(basic.inner, decoded.inner);
    dbg!(decoded);
}

#[derive(Endecode, Debug)]
struct StructCustom<A: Allocator> {
    a:     u32,
    inner: Foo<A>,
}

#[derive(Endecode, Debug, d::PartialEq, d::Eq)]
struct Foo<A>
where
    A: Allocator,
{
    cool:   Box<u64, A>,
    buf:    Box<[u8], A>,
    #[partial_eq(skip)]
    #[endecode(skip)]
    _alloc: A,
}

impl<A: Allocator + Copy> Foo<A> {
    fn new_in(alloc: A) -> Self {
        let mut data = Vec::new_in(alloc);
        data.resize(1024, 0);
        data.iter_mut().enumerate().for_each(|(i, x)| *x = i as u8);

        Foo {
            cool:   Box::new_in(3, alloc),
            buf:    data.into_boxed_slice(),
            _alloc: alloc,
        }
    }

    fn empty(alloc: A) -> Self {
        Foo {
            cool:   Box::new_in(0, alloc),
            buf:    Box::new_in([], alloc),
            _alloc: alloc,
        }
    }
}

#[test]
fn test_struct_with_custom_alloc() {
    let mut data = vec![0u8; 1024];
    data.iter_mut().enumerate().for_each(|(i, x)| *x = i as u8);
    let mut a = StructCustom {
        a:     42,
        inner: Foo::new_in(Global),
    };
    let mut b = StructCustom {
        a:     0,
        inner: Foo::empty(Global),
    };
    let mut buf = vec![0u8; 2048].into_boxed_slice();
    let _ = encode(&mut a, buf.as_mut(), Global).unwrap();
    decode_in_place(&mut b, &buf, Global).unwrap();
    assert_eq!(a.a, b.a);
    assert_eq!(a.inner.cool, b.inner.cool);
    assert_eq!(a.inner.buf, b.inner.buf);
    assert_eq!(a.inner.buf, b.inner.buf);
}

#[derive(Endecode, Debug)]
struct Newtype<A: Allocator>(Box<u64, A>, #[endecode(skip)] A);

#[test]
fn test_newtype() {
    let mut a = Newtype(Box::new(42), Global);
    let mut b = Newtype(Box::new(0), Global);
    let sink = encode(&mut a, Vec::new(), Global).unwrap();
    decode_in_place(&mut b, sink.bytes(), Global).unwrap();
    assert_eq!(a.0, b.0);
}

#[derive(Endecode, Debug, d::PartialEq, Eq)]
#[endecode(alloc(default))]
enum TestEnum<A: Allocator> {
    Foo,
    Bar(u32),
    Baz(#[endecode(alloc(bounds = "copy"))] Foo<A>),
    #[endecode(alloc(default))]
    Fourth {
        #[endecode(alloc(with = new_in))]
        thing: Foo<A>,
    },
}

#[test]
fn test_enum_endecode_same_variant() {
    {
        let mut a = TestEnum::Foo;
        let mut b = TestEnum::Foo;
        let sink = encode(&mut a, Vec::new(), Global).unwrap();
        decode_in_place(&mut b, sink.bytes(), Global).unwrap();
        assert_eq!(a, b);
    }
    {
        let mut a = TestEnum::Bar(42);
        let mut b = TestEnum::Bar(0);
        let sink = encode(&mut a, Vec::new(), Global).unwrap();
        decode_in_place(&mut b, sink.bytes(), Global).unwrap();
        assert_eq!(a, b);
    }
    {
        let mut a = TestEnum::Baz(Foo::new_in(Global));
        let mut b = TestEnum::Baz(Foo::empty(Global));
        let sink = encode(&mut a, Vec::new(), Global).unwrap();
        decode_in_place(&mut b, sink.bytes(), Global).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn test_enum_endecode_different_variant() {
    let mut a = TestEnum::Baz(Foo::new_in(Global));
    let mut b = TestEnum::Foo;
    let sink = encode(&mut a, Vec::new(), Global).unwrap();
    decode_in_place(&mut b, sink.bytes(), Global).unwrap();
    assert_eq!(a, b);
}

#[derive(Endecode, Debug, d::PartialEq, Eq)]
#[endecode(alloc(default))]
enum TestEnumGeneric<A: Allocator, T: Endecode<A> + Default> {
    Foo,
    Bar(u32),
    Baz(#[endecode(alloc(bounds = "copy"))] Foo<A>),
    #[endecode(alloc(default))]
    Fourth {
        #[endecode(alloc(default))]
        thing: T,
    },
}

#[derive(Endecode, Debug, d::PartialEq, Eq)]
#[endecode(alloc(default))]
struct TestString {
    #[endecode(string)]
    a: String,
    #[endecode(string)]
    b: String,
    #[endecode(string)]
    c: String,
    #[endecode(string)]
    d: String,
    e: u64,
}

#[derive(Endecode, Debug, d::PartialEq, Eq)]
#[endecode(alloc(default, global))]
struct TestStringCollectionGlobal {
    #[endecode(alloc(global))]
    strings: Vec<TestString>,
}

#[derive(Endecode, Debug, PartialEq, Default)]
struct Floats {
    a: f32,
    b: f64,
}

#[test]
fn test_floats() {
    let mut a = Floats { a: 0.8, b: 39.3 };
    let mut b = Floats::default();
    let sink = encode(&mut a, Vec::new(), Global).unwrap();
    decode_in_place(&mut b, sink.bytes(), Global).unwrap();
    assert_eq!(a.a, b.a);
    assert_eq!(a.b, b.b);
}

#[derive(Endecode, Debug, PartialEq, Default)]
struct ComplexTuple {
    a: Option<(f32, f32)>,
}

#[derive(Endecode, Debug, PartialEq, Default)]
#[endecode(alloc(default))]
enum TestEnumAllocless {
    #[default]
    #[endecode(alloc(default))]
    A,
    B,
    C,
}

#[derive(Endecode, Debug, PartialEq)]
struct ConstParams<const N: usize, const B: usize> {
    arr: [u32; N],
}

#[derive(Endecode, Debug, PartialEq)]
struct HeaplessVecTest {
    vec: heapless::Vec<Basic, 10>,
}

#[derive(Endecode, Debug, PartialEq)]
struct Bounds<T: Clone> {
    a: T,
    b: T,
}

#[derive(Endecode, Debug, PartialEq)]
struct VramTest {
    #[endecode(alloc(global))]
    buf: Box<[u16]>,
}

#[test]
fn test_vram() {
    let mut data = vec![0u16; 1024];
    data.iter_mut().enumerate().for_each(|(i, x)| *x = i as u16);
    let mut a = VramTest {
        buf: data.into_boxed_slice(),
    };
    let mut b = VramTest { buf: Box::new([]) };

    let sink = vec![0u8; 2048];
    let sink = encode(&mut a, sink, Global).unwrap();
    decode_in_place(&mut b, sink.bytes(), Global).unwrap();

    assert_eq!(a.buf, b.buf)
}
