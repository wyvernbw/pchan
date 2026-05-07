#![doc = include_str!("../README.md")]

pub mod action;
// pub mod animation;
pub mod delete;
pub mod display;
pub mod error;
#[cfg(any(test, feature = "crossterm-tokio"))]
pub mod event_stream;
pub mod image;
pub mod medium;
pub mod tmux;

use std::{error::Error, fmt::Display, io::Write, num::NonZeroU32, time::Duration};

use image::{Image, read_parse_response, read_parse_response_async};

/// Utility trait for allowing us to actually correctly use [`Write::write_all`] on base64
/// encoding. The base64 crate's implementation of [`Write`] for their writer type didn't work
/// correctly with the std `write_all` fn, and it seemed (?) that that was due to some inherent
/// detail about how different operating systems can sometimes write 0 bytes in a single write
/// call, but still be ready for more bytes to be written? Idk. I don't remember anymore. But this
/// fn allows us to work around that and, although we don't use the `base64` crate anymore, I'm
/// still paranoid and I would be shocked if this has any bad side effects. So we're keeping it.
trait Encoder<W: Write>: Write {
	/// basically just [`Write::write_all`] but don't short-circuit if the underlying writer ever
	/// returns `Ok(0)` for an individual [`Write::write`] call
	///
	/// # Errors
	///
	/// - This can error if the underlying [`Write::write`] call fails
	fn write_all_allow_empty(&mut self, slice: &[u8]) -> std::io::Result<()> {
		let mut total_written = 0;
		while let Some(remaining) = slice.get(total_written..) {
			if remaining.is_empty() {
				break;
			}
			total_written += self.write(remaining)?;
		}
		Ok(())
	}
}

impl<W: Write> Encoder<W> for W {}

/// A way to associate specific key characters with the type of integer that correctly represents
/// their range of allowable values
trait Encodable<W: Write, const KEY: char>: PartialEq + Sized {
	/// Their default value - if an integer is equal to this when we're going to write it, we just
	/// don't write anything.
	const DEFAULT: Self;

	/// Write the value, along with its key and a preceding command, to the writer, if it doesn't
	/// equal [`Self::DEFAULT`]
	fn write_kv_encoded(&self, mut writer: W) -> std::io::Result<W> {
		if self == &Self::DEFAULT {
			return Ok(writer);
		}
		write!(writer, ",{KEY}=")?;
		self.write_value_to(writer)
	}

	/// just write its value - the method one that does not have a default implementation
	fn write_value_to(&self, writer: W) -> std::io::Result<W>;
}

/// Convenience trait for writing integers to a writer
pub(crate) trait WriteUint: Write + Sized {
	/// Just `u.write_kv_encoded(self)`, but this way we can chain calls.
	fn write_uint<const KEY: char, E: Encodable<Self, KEY>>(self, u: E) -> std::io::Result<Self> {
		u.write_kv_encoded(self)
	}
}

impl<T: Write + Sized> WriteUint for T {}

/// The format of the image data that is being sent
#[derive(PartialEq, Clone, Debug)]
pub enum PixelFormat {
	/// 3 bytes per pixel, with color in the sRGB color space. If you are using the
	/// [`image`](https://crates.io/crates/image) crate, you can easily create conformant data with
	/// the [`RgbImage`](https://docs.rs/image/latest/image/type.RgbImage.html) struct.
	Rgb24(ImageDimensions, Option<Compression>),
	/// 4 bytes per pixel (3 for the color and 1 for the alpha), with color in the sRGB color
	/// space. If you are using the [`image`](https://crates.io/crates/image) crate, you can easily
	/// create conformant data with the
	/// [`RgbaImage`](https://docs.rs/image/latest/image/type.RgbaImage.html) struct
	Rgba32(ImageDimensions, Option<Compression>),
	/// PNG data - you can specify if compression is being used, and if compression is being used
	/// then you must also supply the total size, in bytes, of the compressed data (not how much
	/// space it will decompress to, but rather how much space the terminal emulator should read to
	/// then decode to get the image)
	Png(Option<(Compression, usize)>)
}

impl<W: Write> Encodable<W, 'f'> for PixelFormat {
	const DEFAULT: Self = Self::Rgb24(
		ImageDimensions {
			width: 0,
			height: 0
		},
		None
	);

	fn write_value_to(&self, mut writer: W) -> std::io::Result<W> {
		fn write_w_h_cmp<W: Write>(
			mut writer: W,
			width: u32,
			height: u32,
			cmp: Option<Compression>
		) -> std::io::Result<W> {
			writer = writer
				.write_uint::<SOURCE_WIDTH_KEY, _>(width)?
				.write_uint::<SOURCE_HEIGHT_KEY, _>(height)?;
			writer = cmp.write_kv_encoded(writer)?;
			Ok(writer)
		}

		match self {
			Self::Rgb24(ImageDimensions { width, height }, cmp) => {
				write!(writer, "24")?;
				write_w_h_cmp(writer, *width, *height, *cmp)
			}
			Self::Rgba32(ImageDimensions { width, height }, cmp) => {
				write!(writer, "32")?;
				write_w_h_cmp(writer, *width, *height, *cmp)
			}
			Self::Png(cmp_and_size) => {
				write!(writer, "100")?;
				if let Some((cmp, size)) = cmp_and_size {
					writer = Some(*cmp)
						.write_kv_encoded(writer)?
						.write_uint::<READ_SIZE_KEY, _>(*size)?;
				}
				Ok(writer)
			}
		}
	}
}

/// The dimensions of the image to display. This specifies a specific crop of the image to display,
/// and must not specify a size larger than the source image. If it is smaller, only a cropped
/// subset of the original image will be sent/displayed
#[derive(PartialEq, Clone, Debug)]
pub struct ImageDimensions {
	/// The number of pixels wide, divided by 4 (for some reason)
	pub width: u32,
	/// The number of pixels high, divided by 4 (for some reason)
	pub height: u32
}

pub(crate) const SOURCE_WIDTH_KEY: char = 's';
pub(crate) const SOURCE_HEIGHT_KEY: char = 'v';
pub(crate) const READ_SIZE_KEY: char = 'S';
pub(crate) const READ_OFFSET_KEY: char = 'O';
pub(crate) const REMAINING_CHUNKS_KEY: char = 'm';
pub(crate) const TRANSFER_ID_KEY: char = 'i';
pub(crate) const IMAGE_NUMBER_KEY: char = 'I';
pub(crate) const PLACEMENT_ID_KEY: char = 'p';
pub(crate) const PIXEL_X_OFFSET_KEY: char = 'X';
pub(crate) const PIXEL_Y_OFFSET_KEY: char = 'Y';
pub(crate) const RELATIVE_HORIZ_CELL_OFFSET_KEY: char = 'H';
pub(crate) const RELATIVE_VERT_CELL_OFFSET_KEY: char = 'V';

pub(crate) const DISPLAY_START_X_KEY: char = 'x';
pub(crate) const DISPLAY_START_Y_KEY: char = 'y';
pub(crate) const DISPLAY_WIDTH_SLICE_KEY: char = 'w';
pub(crate) const DISPLAY_HEIGHT_SLICE_KEY: char = 'h';

pub(crate) const DISPLAY_COLS_KEY: char = 'c';
pub(crate) const DISPLAY_ROWS_KEY: char = 'r';

pub(crate) const Z_INDEX_KEY: char = 'z';

pub(crate) const CURSOR_MOVEMENT_POLICY_KEY: char = 'C';
pub(crate) const VIRTUAL_PLACEMENT_KEY: char = 'U';
pub(crate) const PARENT_ID_KEY: char = 'P';
pub(crate) const PARENT_PLACEMENT_KEY: char = 'Q';

pub(crate) const VERBOSITY_LEVEL_KEY: char = 'q';

/// Implement `Encodable<$key> for $int` for each $key
macro_rules! impl_encodable_for_int {
	($int:ty => $($key:expr,)+) => {
		$(impl<W: Write> Encodable<W, $key> for $int {
			const DEFAULT: Self = 0;
			fn write_value_to(&self, mut writer: W) -> std::io::Result<W> {
				write!(writer, "{self}").map(|()| writer)
			}
		})*
	}
}

impl_encodable_for_int!(i32 => Z_INDEX_KEY, RELATIVE_HORIZ_CELL_OFFSET_KEY, RELATIVE_VERT_CELL_OFFSET_KEY, );
impl_encodable_for_int!(u8 => REMAINING_CHUNKS_KEY, VERBOSITY_LEVEL_KEY,);
impl_encodable_for_int!(u16 => DISPLAY_COLS_KEY, DISPLAY_ROWS_KEY,);
impl_encodable_for_int!(
	u32 => SOURCE_WIDTH_KEY, SOURCE_HEIGHT_KEY, TRANSFER_ID_KEY, IMAGE_NUMBER_KEY,
		   PLACEMENT_ID_KEY, DISPLAY_START_X_KEY, DISPLAY_START_Y_KEY,
		   DISPLAY_WIDTH_SLICE_KEY, DISPLAY_HEIGHT_SLICE_KEY, PARENT_ID_KEY, PARENT_PLACEMENT_KEY,
);
impl_encodable_for_int!(
	usize => READ_SIZE_KEY, READ_OFFSET_KEY, PIXEL_X_OFFSET_KEY, PIXEL_Y_OFFSET_KEY,
);

/// Which method of compression was used to compress the data being sent to the terminal
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Compression {
	/// [RFC 1950](https://datatracker.ietf.org/doc/html/rfc1950.html) zlib based deflate
	/// compression. You can encode data to this format using the
	/// [`flate2`](https://crates.io/crates/flate2) crate's
	/// [`ZlibEncoder`](https://docs.rs/flate2/latest/flate2/write/struct.ZlibEncoder.html)
	ZlibDeflate
}

impl<W: Write> Encodable<W, 'o'> for Option<Compression> {
	const DEFAULT: Self = None;
	fn write_value_to(&self, mut writer: W) -> std::io::Result<W> {
		let Some(compression) = self else {
			return Ok(writer);
		};

		match compression {
			Compression::ZlibDeflate => write!(writer, "z")?
		}

		Ok(writer)
	}
}

/// A trait to facilitate reading from stdin in an async manner - the async version of
/// [`InputReader`]
pub trait AsyncInputReader {
	/// The error type that can occur while trying to read
	type Error: Error;
	/// Read a string from stdin into `buf`. The read-in string should contain all the bytes that
	/// flowed through stdin between (and not including) the first-seen `\x1b` and next-seen
	/// `\x1b`. All non-utf8 input should be discarded.
	///
	/// The attempt to read this input should also be kept to the given timeout - if this function
	/// does not see two `\x1b`s by the time the timeout is hit, it must return an error.
	fn read_esc_delimited_str_with_timeout(
		&mut self,
		buf: &mut String,
		timeout: Duration
	) -> impl Future<Output = Result<(), Self::Error>>;
}

/// A trait to facilitate reading from stdin - the sync version of [`AsyncInputReader`]
pub trait InputReader {
	/// The error type that can occur while trying to read
	type Error: Error;
	/// Does the same as [`AsyncInputReader::read_esc_delimited_str_with_timeout`], except with no
	/// timeout - it is expected to block forever if it can't find `\x1b`s.
	///
	/// # Errors
	///
	/// This can return an error for any reason, depending on the implementation
	fn read_esc_delimited_str(&mut self, buf: &mut String) -> Result<(), Self::Error>;
}

/// Used to either specify an Image ID or Image Number - see the details on [`IdentifierType`] for
/// explanations of how Image IDs vs Numbers are used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumberOrId {
	/// An Image Number ([`IdentifierType::ImageNumber`])
	Number(NonZeroU32),
	/// An Image Id ([`IdentifierType::ImageId`])
	Id(NonZeroU32)
}

/// Sometimes when reading a response from the protocol, we expect the terminal to give us a value
/// back. Sometimes we expect that value to be something specific (e.g. `1`), and sometimes we
/// don't care what the value is - we just want something. This enum is used to distinguish between
/// the two
#[derive(PartialEq, Debug)]
pub enum AnyValueOrSpecific<T> {
	/// It could be any value - we don't really care what
	Any,
	/// We expect it to be this specific value
	Specific(T)
}

impl<T: Display> Display for AnyValueOrSpecific<T> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Any => write!(f, "<any value>"),
			Self::Specific(val) => write!(f, "{val}")
		}
	}
}

/// Multiple types of identifiers exist within the kitty protocol, and this enum just exists to
/// distinguish between them where they perform similar operations
#[derive(Debug, PartialEq)]
pub enum IdentifierType {
	/// A [`NonZeroU32`] relating to a specific image - image IDs can be used to correlate between
	/// an image transmitted to the terminal and an image later displayed on the terminal. Using
	/// the same image ID between multiple operations ensures that they correlate to the same image
	ImageId,
	/// The official protocol documentation explains this best. It states the following:
	///
	/// If you are writing a program that is going to share the screen with other programs and you still want to use image ids, it is not possible to know what image ids are free to use. In this case, instead of using the i key to specify an image id use the I key to specify an image number instead. These numbers are not unique. When creating a new image, even if an existing image has the same number a new one is created. And the terminal will reply with the id of the newly created image. For example, when creating an image with I=13, the terminal will send the response:
	///
	/// ```sh
	/// <ESC>_Gi=99,I=13;OK<ESC>\
	/// ```
	///
	/// Here, the value of `i` is the id for the newly created image and the value of `I` is the same as was sent in the creation command.
	///
	/// All future commands that refer to images using the image number, such as creating placements or deleting images, will act on only the newest image with that number. This allows the client program to send a bunch of commands dealing with an image by image number without waiting for a response from the terminal with the image id. Once such a response is received, the client program should use the i key with the image id for all future communication.
	///
	/// > Note:
	/// > Specifying both i and I keys in any command is an error. The terminal must reply with an EINVAL error message, unless silenced.
	ImageNumber,
	/// To quote the specification:
	///
	/// Since there can be many placements per image, you can also give placements an id. To do so add the p key with a number between 1 and 4294967295. When you specify a placement id, it will be added to the acknowledgement code above. Every placement is uniquely identified by the pair of the image id and the placement id. If you specify a placement id for an image that does not have an id (i.e. has id=0), it will be ignored. In particular this means there can exist multiple images with image id=0, placement id=0. Not specifying a placement id or using p=0 for multiple put commands (a=p) with the same non-zero image id results in multiple placements the image.
	///
	/// An example response:
	///
	/// ```sh
	/// <ESC>_Gi=<image id>,p=<placement id>;OK<ESC>\
	/// ```
	///
	/// If you send two placements with the same image id and placement id the second one will replace the first. This can be used to resize or move placements around the screen, without flicker.
	PlacementId
}

/// An Id that can be used to identify a specific placement of an image so that you can have a
/// single image displayed multiple times on the same surface. See [`IdentifierType::PlacementId`]
pub type PlacementId = NonZeroU32;

/// An Id that can be used to identify an image sent to the terminal - see
/// [`IdentifierType::ImageId`]
pub type ImageId = NonZeroU32;

/// A number that can be used to identify an image sent to the terminal - see
/// [`IdentifierType::ImageNumber`]
pub type ImageNumber = NonZeroU32;

/// A way to configure how the terminal responds to all the commands written to it
#[derive(Clone, Copy, PartialEq, Hash, Debug, Default)]
pub enum Verbosity {
	/// The terminal should respond with an OK or error code to everything
	#[default]
	All = 0,
	/// The terminal should respond with an error code if anything went wrong, but should otherwise
	/// be silent
	ErrorsOnly = 1,
	/// The terminal should not respond at all, regardless of if your operation succeeds or fails
	Silent = 2
}

#[cfg(test)]
pub(crate) mod lib_tests {
	use std::{
		convert::Infallible,
		hash::{DefaultHasher, Hasher as _},
		num::NonZeroU16,
		path::{Path, PathBuf},
		process::{Command, Stdio},
		time::{SystemTime, UNIX_EPOCH}
	};

	use ::image::ImageReader;
	use flate2::{Compression as FlateCompression, write::ZlibEncoder};
	use nix::{sys::stat::Mode, unistd::mkfifo};

	use super::*;
	use crate::{
		Compression,
		action::Action,
		display::DisplayConfig,
		error::{TerminalError, TransmitError},
		image::{Image, parse_response},
		medium::{ChunkSize, Medium}
	};

	fn spawn_kitty_get_io(input: &[u8]) -> String {
		let mut hasher = DefaultHasher::new();
		hasher.write(input);
		let val = hasher.finish();

		let now = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap()
			.as_nanos()
			& u128::from(u16::MAX);

		let fifo_path = format!("/tmp/.kittyimg_test_pipe_{val}_{now}");
		_ = std::fs::remove_file(&fifo_path);
		mkfifo(
			fifo_path.as_str(),
			Mode::S_IRWXU | Mode::S_IRWXG | Mode::S_IRWXO
		)
		.unwrap();

		let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kitty_test.sh");

		let mut cmd = Command::new("kitty")
			.args(["--start-as", "hidden", "bash"])
			.arg(script_path)
			.env("KITTYIMG_PIPE", fifo_path.as_str())
			.stdout(Stdio::null())
			.stdin(Stdio::null())
			.stderr(Stdio::null())
			.spawn()
			.unwrap();

		std::fs::write(&fifo_path, input).unwrap();
		let s = std::fs::read_to_string(fifo_path).unwrap();

		let status = cmd.wait().unwrap();

		assert!(status.success());
		s
	}

	fn spawn_kitty_with_image(image: Image<'_>) -> Result<(), TransmitError<Infallible>> {
		let mut output = Vec::new();
		let num_or_id = image.num_or_id;
		let action = Action::TransmitAndDisplay {
			image,
			config: DisplayConfig::default(),
			placement_id: None
		};

		action
			.write_transmit_to(&mut output, Verbosity::All)
			.unwrap();

		println!("here's output: {:?}", str::from_utf8(&output).unwrap());

		let response = spawn_kitty_get_io(&output);

		parse_response(response, num_or_id, None).map_or_else(
			|e| Err(TransmitError::ParsingResponse(e)),
			|res| res.map_err(TransmitError::Terminal)
		)?;

		drop(action);

		Ok(())
	}

	pub fn png_path() -> Box<Path> {
		PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.join("kitty.png")
			.into()
	}

	#[tokio::test]
	async fn basic_functionality() {
		// TODO: I still don't understand how the image dimensions work tbh. This image is 256x256
		// pixels but 50 is the maximum number we can put for either dimension to make it work. And
		// that changed after we losslessly compressed it - it used to be able to work with 64x64.
		// idk.
		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgb24(
				ImageDimensions {
					width: 50,
					height: 50
				},
				None
			),
			medium: Medium::File(png_path())
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn correctly_fails() {
		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgb24(
				ImageDimensions {
					width: 100,
					height: 100
				},
				None
			),
			medium: Medium::Direct {
				data: (&[]).into(),
				chunk_size: None
			}
		};

		let err = spawn_kitty_with_image(img).unwrap_err();
		assert_eq!(
			err,
			TransmitError::Terminal(TerminalError::NoData("Insufficient image data".into()))
		);
	}

	#[tokio::test]
	async fn direct_unchunked_rgb24_succeeds() {
		let img_data = ImageReader::open(png_path())
			.unwrap()
			.decode()
			.unwrap()
			.to_rgb8();
		dbg!(&img_data);

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgb24(
				ImageDimensions {
					width: img_data.width(),
					height: img_data.height()
				},
				None
			),
			medium: Medium::Direct {
				data: img_data.as_raw().into(),
				chunk_size: None
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn direct_chunked_rgb24_succeeds() {
		let img_data = ImageReader::open(png_path())
			.unwrap()
			.decode()
			.unwrap()
			.to_rgb8();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgb24(
				ImageDimensions {
					width: img_data.width(),
					height: img_data.height()
				},
				None
			),
			medium: Medium::Direct {
				data: img_data.as_raw().into(),
				chunk_size: Some(ChunkSize::new(NonZeroU16::new(32).unwrap()).unwrap())
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn direct_unchunked_compressed_rgb24_succeeds() {
		let img_data = ImageReader::open(png_path())
			.unwrap()
			.decode()
			.unwrap()
			.to_rgb8();
		dbg!(&img_data);
		let dim = ImageDimensions {
			width: img_data.width(),
			height: img_data.height()
		};

		let mut encoder = ZlibEncoder::new(Vec::new(), FlateCompression::fast());
		encoder.write_all(&img_data).unwrap();
		let compressed = encoder.finish().unwrap();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgb24(dim, Some(Compression::ZlibDeflate)),
			medium: Medium::Direct {
				data: compressed.into(),
				chunk_size: None
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn direct_unchunked_rgba32_succeeds() {
		let img_data = ImageReader::open(png_path())
			.unwrap()
			.decode()
			.unwrap()
			.to_rgba8();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgba32(
				ImageDimensions {
					width: img_data.width(),
					height: img_data.height()
				},
				None
			),
			medium: Medium::Direct {
				data: img_data.as_raw().into(),
				chunk_size: None
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn direct_chunked_rgba32_succeeds() {
		let img_data = ImageReader::open(png_path())
			.unwrap()
			.decode()
			.unwrap()
			.to_rgba8();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Rgba32(
				ImageDimensions {
					width: img_data.width(),
					height: img_data.height()
				},
				None
			),
			medium: Medium::Direct {
				data: img_data.as_raw().into(),
				chunk_size: Some(ChunkSize::new(NonZeroU16::new(132).unwrap()).unwrap())
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn direct_unchunked_png_succeeds() {
		let img_data = std::fs::read(png_path()).unwrap();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Png(None),
			medium: Medium::Direct {
				data: img_data.into(),
				chunk_size: None
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn direct_unchunked_compressed_png_succeeds() {
		let img_data = std::fs::read(png_path()).unwrap();

		let precompressed_size = img_data.len();
		let mut compressor = ZlibEncoder::new(Vec::new(), FlateCompression::fast());
		compressor.write_all(&img_data).unwrap();
		let img_data = compressor.finish().unwrap();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Png(Some((Compression::ZlibDeflate, precompressed_size))),
			medium: Medium::Direct {
				data: img_data.into(),
				chunk_size: None
			}
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	async fn file_path_png_succeeds() {
		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Png(None),
			medium: Medium::File(png_path())
		};

		spawn_kitty_with_image(img).unwrap();
	}

	#[tokio::test]
	#[cfg(unix)]
	async fn png_unix_shm_succeeds() {
		use crate::medium::SharedMemObject;

		let data = std::fs::read(png_path()).unwrap();
		let name = format!(
			"__kitty_img_test_{}",
			std::time::SystemTime::now()
				.duration_since(UNIX_EPOCH)
				.unwrap()
				.as_nanos()
		);

		let mut obj = SharedMemObject::create_new(&name, data.len()).unwrap();
		obj.copy_in_buf(&data).unwrap();

		let img = Image {
			num_or_id: NumberOrId::Id(NonZeroU32::new(1).unwrap()),
			format: PixelFormat::Png(None),
			medium: Medium::SharedMemObject(obj)
		};

		spawn_kitty_with_image(img).unwrap();
	}
}
