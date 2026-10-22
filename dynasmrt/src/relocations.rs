//! This module defines the `Relocation` trait and several utilities for implementing relocations.

use byteorder::{ByteOrder, LittleEndian};
use std::fmt::Debug;

use std::convert::TryFrom;

/// Error returned when encoding a relocation failed
#[derive(Debug)]
pub struct ImpossibleRelocation { }


/// Used to inform assemblers on how to implement relocations for each architecture.
/// When implementing a new architecture, one simply has to implement this trait for
/// the architecture's relocation definition.
pub trait Relocation {
    /// construct this relocation from an encoded representation.
    fn from_encoding(encoding: u8) -> Self;
    /// construct this relocation from a simple size. This is used to implement relocations in directives and literal pools.
    fn from_size(kind: RelocationKind, size: RelocationSize) -> Self;
    /// The size of the slice of bytes affected by this relocation
    fn size(&self) -> usize;
    /// Write a value into a buffer of size `self.size()` in the format of this relocation.
    /// Any bits not part of the relocation should be preserved.
    fn write_value(&self, buf: &mut [u8], value: isize) -> Result<(), ImpossibleRelocation>;
    /// Read a value from a buffer of size `self.size()` in the format of this relocation.
    fn read_value(&self, buf: &[u8]) -> isize;
    /// Specifies what kind of relocation this relocation instance is.
    fn kind(&self) -> RelocationKind;
    /// Specifies the default page size on this platform.
    fn page_size() -> usize;
}

/// Trait that just handles the internal decoding of architecture-specific relocation encodings
/// This is useful so the relocation type machinery can be generic over it.
pub trait ArchitectureRelocationEncoding : Clone + Copy + Debug {
    /// decodes this custom relocation encoding from the bitfield in the relocation code
    fn decode(code: u8) -> Self;
}


/// Enum that specifies if/how relocations should be adapted if the assembling buffer is moved
#[derive(Clone, Copy, Debug)]
pub enum RelocationKind {
    /// A simple, PC-relative relocation. These can be encoded once and do not need
    /// to be adjusted when the executable buffer is moved.
    Relative,
    /// An absolute relocation to a relative address,
    /// i.e. trying to put the address of a dynasm x86 function in a register
    /// This means adjustment is necessary when the executable buffer is moved
    AbsToRel,
    /// A relative relocation to an absolute address,
    /// i.e. trying to call a Rust function with a dynasm x86 call.
    /// This means adjustment is necessary when the executable buffer is moved
    RelToAbs,
    /// An absolute relocation to an absolute address
    /// This isn't particularly useful, but the user can specify it sometimes
    Absolute
}

/// Enum that specifies how a certain relocation is encoded.
#[derive(Clone, Copy, Debug)]
pub enum RelocationEncoding<A: ArchitectureRelocationEncoding> {
    /// This relocation is just some bytes of data
    Simple(RelocationSize),
    /// This relocation is complex and requires architecture-specific decoding logic
    ArchSpecific(A)
}

/// `RelocationType` contains all information needed to describe how a relocation should be
/// performed by the runtime
#[derive(Clone, Copy, Debug)]
pub struct RelocationType<A: ArchitectureRelocationEncoding> {
    /// How this relocation should be adapted if the buffer gets moved
    pub kind: RelocationKind,
    /// The way this relocation is to be encoded
    pub encoding: RelocationEncoding<A>
}

impl<A: ArchitectureRelocationEncoding> RelocationType<A> {
    /// decode the packed representation emitted by the plugin
    pub fn decode(code: u8) -> Self {
        let kind = match code >> 6 {
            0 => RelocationKind::Relative,
            1 => RelocationKind::AbsToRel,
            2 => RelocationKind::RelToAbs,
            3 => RelocationKind::Absolute,
            _ => unreachable!()
        };

        let encoding = match code & 0x3F {
            0 => RelocationEncoding::Simple(RelocationSize::Byte),
            1 => RelocationEncoding::Simple(RelocationSize::Word),
            2 => RelocationEncoding::Simple(RelocationSize::DWord),
            3 => RelocationEncoding::Simple(RelocationSize::QWord),
            c => RelocationEncoding::ArchSpecific(A::decode(c - 4))
        };
        RelocationType {
            kind,
            encoding
        }
    }

    /// manually create a `RelocationType` for a simple size-based relocation
    pub fn from_size(kind: RelocationKind, size: RelocationSize) -> Self {
        RelocationType {
            kind,
            encoding: RelocationEncoding::Simple(size)
        }
    }
}

/// A simple size-based relocation descriptor for relocations in data directives.
/// Can be converted to a relocation for any kind of architecture using `Relocation::from_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelocationSize {
    /// A byte-sized relocation
    Byte = 1,
    /// A two-byte relocation
    Word = 2,
    /// A four-byte sized relocation
    DWord = 4,
    /// An 8-byte sized relocation
    QWord = 8,
}

impl RelocationSize {
    /// The size of this size-based relocation in bytes
    pub fn size(&self) -> usize {
        *self as usize
    }

    /// Pack `value` into this relocation size and format it into `buf`
    pub fn write_value(&self, buf: &mut [u8], value: isize) -> Result<(), ImpossibleRelocation> {
        match self {
            RelocationSize::Byte => buf[0] =
                i8::try_from(value).map_err(|_| ImpossibleRelocation { } )?
            as u8,
            RelocationSize::Word => LittleEndian::write_i16(buf,
                i16::try_from(value).map_err(|_| ImpossibleRelocation { } )?
            ),
            RelocationSize::DWord => LittleEndian::write_i32(buf,
                i32::try_from(value).map_err(|_| ImpossibleRelocation { } )?
            ),
            RelocationSize::QWord => LittleEndian::write_i64(buf,
                i64::try_from(value).map_err(|_| ImpossibleRelocation { } )?
            ),
        }
        Ok(())
    }

    /// Extract a value of this size from `buf`
    pub fn read_value(&self, buf: &[u8]) -> isize {
        match self {
            RelocationSize::Byte => buf[0] as i8 as isize,
            RelocationSize::Word => LittleEndian::read_i16(buf) as isize,
            RelocationSize::DWord => LittleEndian::read_i32(buf) as isize,
            RelocationSize::QWord => LittleEndian::read_i64(buf) as isize,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum NoComplexRelocationEncodings {}

impl ArchitectureRelocationEncoding for NoComplexRelocationEncodings {
    fn decode(code: u8) -> Self {
        panic!("Invalid complex relocation code {code} given for the current architecture");
    }
}

/// A simple relocation type for relocations that do not need complex bitpacking.
#[derive(Debug, Clone, Copy)]
pub struct SimpleRelocation(RelocationType<NoComplexRelocationEncodings>);

/// A relocation that has no architecture-specific encoding/decoding logic
impl Relocation for SimpleRelocation {
    fn from_encoding(encoding: u8) -> Self {
        SimpleRelocation(RelocationType::decode(encoding))
    }

    fn from_size(kind: RelocationKind, size: RelocationSize) -> Self {
        SimpleRelocation(RelocationType::from_size(kind, size))
    }

    fn size(&self) -> usize {
        match self.0.encoding {
            RelocationEncoding::Simple(s) => s.size(),
            _ => unreachable!()
        }
    }

    fn write_value(&self, buf: &mut [u8], value: isize) -> Result<(), ImpossibleRelocation> {
        match self.0.encoding {
            RelocationEncoding::Simple(s) => s.write_value(buf, value),
            _ => unreachable!()
        }
    }

    fn read_value(&self, buf: &[u8]) -> isize {
        match self.0.encoding {
            RelocationEncoding::Simple(s) => s.read_value(buf),
            _ => unreachable!()
        }
    }

    fn kind(&self) -> RelocationKind {
        self.0.kind
    }

    fn page_size() -> usize {
        4096
    }
}

pub(crate) fn fits_signed_bitfield(value: i64, bits: u8) -> bool {
    if bits >= 64 {
        return true;
    }

    let half = 1i64 << (bits - 1);
    value < half && value >= -half
}
