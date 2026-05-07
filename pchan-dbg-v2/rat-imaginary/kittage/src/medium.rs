//! The [`Medium`], [`SharedMemObject`] and [`ChunkSize`] structs for configuring how data is sent

use std::{borrow::Cow, io::Write, num::NonZeroU16, path::Path};

use base64_simd::{Base64, Out};

use crate::Encoder as _;

/// The amount of data that should be sent within a single escape code when transferring data via a
/// 'direct' medium (see [`Medium::Direct`]) - this data does not necessarily *need* to be chunked
/// when sending directly, but not chunking it increases the likelihood that the transferred image
/// will be rejected, as escape codes have maximum lengths on many platforms. See
/// [`ChunkSize::new`] for specifications of what it should contain
//
// INVARIANT: `self.0` must always be <= 1024
#[derive(Debug, PartialEq)]
pub struct ChunkSize(NonZeroU16);

impl ChunkSize {
	/// `size` represents the number of 4-byte chunks of (already base64-encoded) data sent within
	/// each escape code. If it is greater than 1024, this will return [`None`] as the protocol
	/// specifies that no more than 4096 bytes may be sent at a time.
	#[must_use]
	pub fn new(size: NonZeroU16) -> Option<Self> {
		(size.get() <= 1024).then_some(Self(size))
	}
}

impl Default for ChunkSize {
	fn default() -> Self {
		Self(const { NonZeroU16::new(1024).unwrap() })
	}
}

/// The medium through which the file itself will be transferred to the terminal
#[derive(Debug, PartialEq)]
pub enum Medium<'data> {
	/// Direct (the data is transmitted within the escape code itself)
	Direct {
		/// The maximum length of data sent within each escape code. To quote from the
		/// specification:
		///
		/// > Remote clients, those that are unable to use the filesystem/shared memory to transmit data, must send the pixel data directly using escape codes. Since escape codes are of limited maximum length, the data will need to be chunked up for transfer.
		///
		/// See the documentation on [`ChunkSize`] for more details
		chunk_size: Option<ChunkSize>,
		/// The image data to be displayed - when using this to display an [`Image`] (or within an
		/// [`Action`]), this data must be of the same underlying image format as the
		/// [`PixelFormat`] specified.
		///
		/// [`Image`]: crate::Image
		/// [`Action`]: crate::action::Action
		/// [`PixelFormat`]: crate::PixelFormat
		data: Cow<'data, [u8]>
	},
	/// A simple file (regular files only, not named pipes, device files, etc.)
	File(Box<Path>),
	/// A temporary file, the terminal emulator will delete the file after reading the pixel data.
	/// For security reasons the terminal emulator should only delete the file if it is in a known
	/// temporary directory, such as `/tmp`, `/dev/shm`, `TMPDIR` env var if present, and any
	/// platform specific temporary directories and the file has the string `tty-graphics-protocol`
	/// in its full file path
	TempFile(Box<Path>),
	/// A _shared memory object_, which on POSIX systems is a [POSIX shared memory
	/// object](https://pubs.opengroup.org/onlinepubs/9699919799/functions/shm_open.html)
	/// and on Windows is a [Named shared memory object](https://docs.microsoft.com/en-us/windows/win32/memory/creating-named-shared-memory).
	SharedMemObject(SharedMemObject)
}

#[cfg(unix)]
#[derive(Debug)]
struct UnixShm {
	/// the shm itself - this must not be touched while this struct exists
	shm: psx_shm::UnlinkOnDrop,
	/// the map that we can use to write to the data held behind this struct
	map: memmap2::MmapMut
}

/// A Shared memory object which can be used for transferring an image over to the terminal. Works
/// on both unix and windows
#[derive(Debug)]
pub struct SharedMemObject {
	/// The shm itself - nested since it needs two fields whereas the windows one only needs one
	#[cfg(unix)]
	inner: UnixShm,

	/// The shm itself
	#[cfg(windows)]
	inner: winmmf::MemoryMappedFile<winmmf::RWLock<'static>>
}

impl PartialEq for SharedMemObject {
	fn eq(&self, other: &Self) -> bool {
		// This is a very inefficient comparison 'cause it requires two allocations but whatever...
		// I'm not sure how else to do it
		self.name() == other.name()
	}
}

/// Failure that can happen when calling [`SharedMemObject::create_new`]; see its documentation
/// for more details
#[cfg(unix)]
#[derive(Debug)]
pub struct ShmError {
	/// The step of [`SharedMemObject::create_new`] that causes `err`
	pub step: ShmCreationFailureStep,
	/// The error caused at the step `step`
	pub err: std::io::Error
}

#[cfg(unix)]
impl std::fmt::Display for ShmError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let Self { step, err } = self;
		match step {
			ShmCreationFailureStep::ShmOpen => write!(f, "Couldn't open SHM: {err}"),
			ShmCreationFailureStep::ShmSetSize =>
				write!(f, "Couldn't set the size of the SHM: {err}"),
			ShmCreationFailureStep::MemmapCreation => write!(
				f,
				"Couldn't create the memmap necessary to write to this SHM: {err}"
			)
		}
	}
}

#[cfg(unix)]
impl core::error::Error for ShmError {}

/// The point at which the error contained in [`ShmError`] happens in
/// [`SharedMemObject::create_new`]
#[cfg(unix)]
#[derive(Debug)]
pub enum ShmCreationFailureStep {
	/// The call to [`psx_shm::Shm::open`] failed
	ShmOpen,
	/// The call to [`psx_shm::Shm::set_size`] failed
	ShmSetSize,
	/// The call to [`psx_shm::Shm::map`] failed
	MemmapCreation
}

impl SharedMemObject {
	/// Construct a new instance from just a name and a size. This opens a completely new shm - the
	/// name must not be in use by any other shm on the system (this function will error if it is).
	///
	/// # Errors
	///
	/// This can return an error if any of the following are true:
	/// - The provided name is already in use by another shm on the system
	/// - An underlying syscall fails for some reason
	///
	/// The [`ShmCreationFailureStep`] that accompanies the [`std::io::Error`] in the result
	/// clarifies exactly which underlying call produced the io error.
	#[cfg(unix)]
	pub fn create_new(name: &str, size: usize) -> Result<Self, ShmError> {
		use psx_shm::UnlinkOnDrop;
		use rustix::{fs::Mode, shm::OFlags};

		let mut shm = psx_shm::Shm::open(
			name,
			OFlags::RDWR | OFlags::CREATE | OFlags::EXCL,
			Mode::RUSR | Mode::WUSR
		)
		.map_err(|err| ShmError {
			step: ShmCreationFailureStep::ShmOpen,
			err
		})?;

		shm.set_size(size).map_err(|err| ShmError {
			step: ShmCreationFailureStep::ShmSetSize,
			err
		})?;

		// SAFETY: We are not mapping an actual file on disc, and because we use the `EXCL` flag up
		// above, we can ensure that no other process has access to this (unless they, immediately
		// after we created the shm, happened to also open it with the same name without us telling
		// them about it, but that's the same sort of risk as someone writing to /proc/mem so we're
		// not worrying about it)
		let borrowed_mmap = unsafe { shm.map(0) }.map_err(|err| ShmError {
			step: ShmCreationFailureStep::MemmapCreation,
			err
		})?;
		// SAFETY: This is sound because we are not then creating another map
		let map = unsafe { borrowed_mmap.into_map() };

		Ok(Self {
			inner: UnixShm {
				shm: UnlinkOnDrop { shm },
				map
			}
		})
	}

	/// Construct a new instance of self on windows, using the given name and size.
	///
	/// # Errors
	///
	/// This can return an error if the inner call to [`winmmf::MemoryMappedFile::new`] fails
	#[cfg(windows)]
	pub fn create_new(
		name: String,
		size: core::num::NonZeroUsize
	) -> Result<Self, winmmf::err::Error> {
		use winmmf::{MemoryMappedFile, RWLock, mmf::Namespace};

		MemoryMappedFile::<RWLock<'static>>::new(size, name, Namespace::LOCAL, None)
			.map(|inner| Self { inner })
	}

	fn name(&self) -> Box<str> {
		#[cfg(unix)]
		{
			self.inner.shm.shm.name().into()
		}

		#[cfg(windows)]
		{
			self.inner.fullname().into()
		}
	}

	/// Replace the entire contents of this shm's memory with the given buffer. This will return an
	/// error if the buffer can't fit in self.
	///
	/// # Errors
	///
	/// On unix, this can only fail if the buffer size provided is larger than the size this shm
	/// was created with.
	///
	/// On windows, this can fail for any reason representable by [`winmmf::err::Error`]
	pub fn copy_in_buf(&mut self, buf: &[u8]) -> std::io::Result<()> {
		let buf_len = buf.len();

		#[cfg(unix)]
		{
			use std::cmp::Ordering;

			match buf_len.cmp(&self.inner.map.as_ref().len()) {
				Ordering::Less => {
					self.inner.map[..buf_len].copy_from_slice(buf);
					for byte in &mut self.inner.map[buf_len..] {
						*byte = 0;
					}
				}
				Ordering::Equal => self.inner.map.copy_from_slice(buf),
				Ordering::Greater =>
					return Err(std::io::Error::new(
						std::io::ErrorKind::InvalidInput,
						"Provided buffer is too large to fit"
					)),
			}

			Ok(())
		}

		#[cfg(windows)]
		{
			use std::io::{Error as IOError, ErrorKind};

			use winmmf::{Mmf as _, err::Error};

			self.inner.write(buf)
				.map_err(|e| match e {
					Error::ReadLocked => IOError::new(
						ErrorKind::ResourceBusy,
						"The MMF is locked by a reader, so we can't write to it at the moment"
					),
					Error::WriteLocked => IOError::new(
						ErrorKind::ResourceBusy,
						"The MMF is locked by another writer, so we can't write to it at the moment"
					),
					Error::Uninitialized => IOError::new(
						ErrorKind::InvalidData,
						"The MMF is uninitialized, cannot write to it. Please recreate it and try again."
					),
					Error::MaxReaders => IOError::new(
						ErrorKind::QuotaExceeded,
						"More than 128 readers were active at the same time; back off"
					),
					Error::NotEnoughMemory => IOError::new(
						ErrorKind::OutOfMemory,
						"The MMF does not have enough memory to store this buffer"
					),
					Error::MMF_NotFound => IOError::other(
						"The MMF was opened as read-only, was already closed, or couldn't be initialized properly"
					),
					Error::LockViolation =>
						IOError::other("A lock violation occurred with this MMF"),
					Error::MaxTriesReached => IOError::new(
						ErrorKind::ResourceBusy,
						"The MMF is locked and we spun for the max amount of times to try to get access without success"
					),

					Error::LargePagePermissionError => IOError::new(
						ErrorKind::PermissionDenied,
						"This process could not acquire the necesssary permissions to use large pages, but this MMF requires large pages."
					),
					Error::GeneralFailure => IOError::other("An inspecific, general error occurred"),
					Error::OS_Err(e) | Error::OS_OK(e) => IOError::other(e)
				})
		}
	}
}

/// Just a wrapper for encoding a chunk of data as base64 and writing it to a writer
pub(crate) fn write_b64<W: Write>(data: &[u8], mut writer: W) -> std::io::Result<W> {
	const BUF_SIZE: usize = 1024;
	let mut buf = [0u8; BUF_SIZE];
	const ENCODER: Base64 = base64_simd::STANDARD_NO_PAD;

	for chunk in data.chunks(const { ENCODER.estimated_decoded_length(BUF_SIZE) }) {
		let encoded_chunk = ENCODER.encode(chunk, Out::from_slice(buf.as_mut_slice()));
		writer.write_all_allow_empty(encoded_chunk)?;
	}

	Ok(writer)
}

impl Medium<'_> {
	/// Write the necessary utf8 for sending this data to kitty into the provided writer, returning
	/// the writer or an error if something goes wrong
	pub(crate) fn write_data<W: Write>(&self, mut writer: W) -> Result<W, std::io::Error> {
		let name: Box<str>;
		let (key, data) = match self {
			Self::Direct { data, chunk_size } => {
				if let Some(chunk_size) = chunk_size {
					write!(writer, ",t=d,")?;
					// spec says we first have to encode, and THEN chunk it. I think that we could
					// do some math and chunk it by like chunk_size * (8 / 6)? or something? but
					// I'll do that later or smth. Either way, before we do that, we have to encode
					// it to a string
					let b64_encoded_bytes_per_chunk = usize::from(chunk_size.0.get()) * 4;
					let pre_encoded_bytes_per_chunk = (b64_encoded_bytes_per_chunk / 4) * 3;
					let total_chunks = data.len().div_ceil(pre_encoded_bytes_per_chunk);

					// then, since we know that it's base64-encoded, we can be confident each
					// character only takes up a single byte, so we can just chunk it by-byte
					let mut chunks = data.chunks(pre_encoded_bytes_per_chunk);

					// we need to write the first one differently since it is just being added onto
					// the end of the current command
					if let Some(first) = chunks.next() {
						write!(writer, "m={};", u8::from(total_chunks != 1))?;
						writer = write_b64(first, writer)?;
						write!(writer, "\x1b\\")?;
					}

					// then all the ones after can just be printed in the same way
					for (idx, chunk) in chunks.enumerate() {
						write!(writer, "\x1b_Gm={};", u8::from(total_chunks != idx + 2))?;
						writer = write_b64(chunk, writer)?;
						write!(writer, "\x1b\\")?;
					}

					return Ok(writer);
				}

				// and if we don't chunk it at all, then just give it to be printed normally
				('d', &**data)
			}
			Self::File(path) => ('f', path.as_os_str().as_encoded_bytes()),
			Self::TempFile(path) => ('t', path.as_os_str().as_encoded_bytes()),
			Self::SharedMemObject(obj) => {
				name = obj.name();
				('s', name.as_bytes())
			}
		};

		write!(writer, ",t={key};")?;
		write_b64(data, writer)
	}
}
