//! The [`Action`] type

use std::io::Write;

use crate::{
	AsyncInputReader, Image, ImageId, InputReader, NumberOrId, PlacementId, Verbosity,
	delete::DeleteConfig, display::DisplayConfig, error::TransmitError, read_parse_response,
	read_parse_response_async
};

/// The different actions one can take to interact with a terminal which supports the kitty image
/// protocol. This is the main interaction point with the terminal - one should construct an
/// [`Action`] and [`Action::execute`] it
#[derive(Debug, PartialEq)]
pub enum Action<'image, 'data> {
	/// This simply sends the image data to the terminal, but does not display it. It also
	/// transfers ownership of the image data to the terminal - for example, if a temp file is used
	/// to send the data, that file will be deleted after it is transmitted to the terminal. Once a
	/// `Transmit` is successfully sent, one can then display the sent image with
	/// [`Action::Display`]
	Transmit(Image<'data>),
	/// Display an image which was already transmitted to (and is now owned by) the terminal
	Display {
		/// The image ID of the image which you want to display
		image_id: ImageId,
		/// A 'placement ID' for this display - see the documentation for [`PlacementId`] for more
		/// info. This must be sent when displaying something that was already transmitted, but is
		/// optional when transmitting and displaying in-one (e.g. with
		/// [`Self::TransmitAndDisplay`]
		placement_id: PlacementId,
		/// The details about exactly how this image should be displayed - the location, cursor
		/// movement, etc
		config: DisplayConfig
	},
	/// Transmit and then display an image. Should act effectively the same as calling
	/// [`Action::Transmit`] and then [`Action::Display`] with the returned image id.
	TransmitAndDisplay {
		/// The image which will be transferred to the terminal and then displayed immediately
		/// after
		image: Image<'data>,
		/// The details about exactly how this image should be displayed - the location, cursor
		/// movement, etc
		config: DisplayConfig,
		/// The placement ID for this display, if you'd like to use one. It is not necessary when
		/// transmitting and displaying in-one
		placement_id: Option<PlacementId>
	},
	/// Query the terminal to determine if a specific image can be transmitted & displayed. The
	/// following is quoted from the spec:
	///
	/// Since a client has no a-priori knowledge of whether it shares a filesystem/shared memory with the terminal emulator, it can send an id with the control data, using the i key (which can be an arbitrary positive integer up to 4294967295, it must not be zero). If it does so, the terminal emulator will reply after trying to load the image, saying whether loading was successful or not. For example:
	///
	/// ```text
	/// <ESC>_Gi=31,s=10,v=2,t=s;<encoded /some-shared-memory-name><ESC>\
	/// ```
	///
	/// to which the terminal emulator will reply (after trying to load the data):
	///
	/// ```text
	/// <ESC>_Gi=31;error message or OK<ESC>\
	/// ```
	///
	/// Here the i value will be the same as was sent by the client in the original request. The message data will be a ASCII encoded string containing only printable characters and spaces. The string will be OK if reading the pixel data succeeded or an error message.
	///
	/// Sometimes, using an id is not appropriate, for example, if you do not want to replace a previously sent image with the same id, or if you are sending a dummy image and do not want it stored by the terminal emulator. In that case, you can use the query action, set a=q. Then the terminal emulator will try to load the image and respond with either OK or an error, as above, but it will not replace an existing image with the same id, nor will it store the image.
	///
	/// We intend that any terminal emulator that wishes to support it can do so. To check if a terminal emulator supports the graphics protocol the best way is to send the above query action followed by a request for the [primary device attributes](https://vt100.net/docs/vt510-rm/DA1.html). If you get back an answer for the device attributes without getting back an answer for the query action the terminal emulator does not support the graphics protocol.
	///
	/// This means that terminal emulators that support the graphics protocol, must reply to query actions immediately without processing other input. Most terminal emulators handle input in a FIFO manner, anyway.
	///
	/// So for example, you could send:
	///
	/// ```text
	/// <ESC>_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA<ESC>\<ESC>[c
	/// ```
	///
	/// If you get back a response to the graphics query, the terminal emulator supports the protocol, if you get back a response to the device attributes query without a response to the graphics query, it does not.
	Query(&'image Image<'data>),
	/// Delete a specific set of images. See the [`DeleteConfig`] documentation for more details
	/// about how to use
	Delete(DeleteConfig) // we'll do these at some point
	                     // TransmitAnimationFrames,
	                     // ControlAnimation,
	                     // ComposeAnimationFrames
}

impl Action<'_, '_> {
	/// Write the transmit code for this [`Action`] to `writer` - this is the first part of
	/// [`Self::execute`] and only does a part of what is necessary to fully interact with a
	/// terminal. The full details can be found at [`Self::execute`].
	///
	/// # Errors
	///
	/// This will error if writing to `writer` ever returns an error.
	pub fn write_transmit_to<W: Write>(
		&self,
		writer: W,
		verbosity: Verbosity
	) -> Result<W, std::io::Error> {
		fn inner_for_stdio<W: Write>(
			img: &Action<'_, '_>,
			mut writer: W,
			verbosity: Verbosity
		) -> Result<W, std::io::Error> {
			write!(writer, "\x1b_Ga=")?;

			let mut writer = match img {
				Action::Transmit(image) => {
					write!(writer, "t,")?;
					image.write_transmit_to(writer, None, None, verbosity)?
				}
				Action::TransmitAndDisplay {
					image,
					config,
					placement_id
				} => {
					write!(writer, "T,")?;
					image.write_transmit_to(writer, *placement_id, Some(config), verbosity)?
				}
				Action::Query(image) => {
					write!(writer, "q,")?;
					image.write_transmit_to(writer, None, None, verbosity)?
				}
				Action::Display {
					image_id,
					placement_id,
					config
				} => {
					write!(writer, "p,i={image_id},p={placement_id}")?;
					writer = config.write_to(writer)?;
					write!(writer, "\x1b\\")?;
					writer
				}
				Action::Delete(del) => {
					write!(writer, "d")?;
					del.write_to(writer)?
				}
			};

			writer.flush()?;

			Ok(writer)
		}

		inner_for_stdio(self, writer, verbosity)
	}

	/// This pulls the [`NumberOrId`] and [`PlacementId`] out of self
	fn extract_num_or_id_and_placement(&self) -> Option<(NumberOrId, Option<PlacementId>)> {
		match self {
			Self::Transmit(img) => Some((img.num_or_id, None)),
			Self::TransmitAndDisplay {
				image,
				placement_id,
				..
			} => Some((image.num_or_id, *placement_id)),
			Self::Query(img) => Some((img.num_or_id, None)),
			Self::Display {
				image_id,
				placement_id,
				..
			} => Some((NumberOrId::Id(*image_id), Some(*placement_id))),
			Self::Delete(_) => None
		}
	}

	/// This is the main point of interaction with this library - to display an image, you need to
	/// create an [`Action`] and then call this function on it.
	///
	/// This function does two main things:
	/// 1. Writes the necessary escape codes to `writer`, then flushes it.
	/// 2. Unlinks the shared memory object if necessary (see [`Medium::SharedMemObject`])
	/// 3. Reads in the terminal's response via `reader`
	/// 4. Parses the terminal's response and returns any errors that occur or are transmitted
	///
	/// Steps 1 & 2 are performed simply by calling [`Self::write_transmit_to`], so this library
	/// can be used in a sans-io method by using that instead. (**TODO**: make the parse method pub
	/// in a more ergonomic API)
	///
	/// For this function to work correctly, `writer` should be writing directly to something that
	/// flushes directly to a kitty-supporting terminal. This function assumes that, once flushed
	/// to `writer`, the terminal will respond and this response can be read by `reader`.
	///
	/// [`Medium::SharedMemObject`]: crate::medium::Medium::SharedMemObject
	///
	/// # Errors
	///
	/// This can return errors if:
	/// - Writing to `writer` fails at any point
	/// - The `reader` fails to read a response from the terminal
	/// - The response from the terminal is unparseable
	/// - The response from the terminal informs us that it ran into an error
	pub fn execute<W: Write, I: InputReader>(
		self,
		writer: W,
		reader: I
	) -> Result<(W, Option<ImageId>), TransmitError<I::Error>> {
		let id_and_p = self.extract_num_or_id_and_placement();

		let writer = self
			.write_transmit_to(writer, Verbosity::All)
			.map_err(TransmitError::Writing)?;

		let img_id = if let Some((id_or_num, placement_id)) = id_and_p {
			Some(read_parse_response(reader, id_or_num, placement_id)?)
		} else {
			None
		};

		Ok((writer, img_id))
	}

	/// An async version of [`Self::execute`] - check its documentation for more details
	///
	/// # Errors
	///
	/// - This may return an error in the same situations that [`Self::execute`] may.
	pub async fn execute_async<W: Write, I: AsyncInputReader>(
		self,
		writer: W,
		reader: I
	) -> Result<(W, Option<ImageId>), TransmitError<I::Error>> {
		let id_and_p = self.extract_num_or_id_and_placement();

		let writer = self
			.write_transmit_to(writer, Verbosity::All)
			.map_err(TransmitError::Writing)?;

		let img_id = if let Some((id_or_num, placement_id)) = id_and_p {
			Some(read_parse_response_async(reader, id_or_num, placement_id).await?)
		} else {
			None
		};

		Ok((writer, img_id))
	}
}
