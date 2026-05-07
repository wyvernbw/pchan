//! Errors that can occur while interacting with this library

use std::{error::Error, num::NonZeroU32};

use crate::{AnyValueOrSpecific, IdentifierType};

/// An error that can occur while parsing a response from the terminal that we send a message to.
/// The exact format of the response is unspecified in the kitty docs, but we try to conform to the
/// examples of correct and wrong responses given in our parsing
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ParseError {
	/// The terminal's response didn't start with the expected sequence
	#[error("Terminal Response didn't start with '\\x1b_G' but instead '{0}'")]
	NoStartSequence(String),
	/// The response had no semicolon (which is used to delimit the options at the beginning from
	/// the response data)
	#[error("No final semicolon was provided to indicate end of options in the terminal response")]
	NoFinalSemicolon,
	/// There was an option before the semicolon which wasn't parseable with the normal
	/// `<char>=<char_or_number>` format
	#[error(
		"A key-value option in a terminal response was unparseable; expected <char>=<char_or_number>"
	)]
	MalformedResponseOption(String),
	/// We expected to receive an ID of the given type in the response, but the response didn't
	/// provide such an ID.
	#[error("An ID of type {ty:?} was sent, but no relevant ID was seen in the terminal response")]
	NoResponseId {
		/// The type of the identifier which we expected to receive
		ty: IdentifierType
	},
	/// We got an ID in the response, but it had a different value than what we were expecting to
	/// get. E.g. We sent a message for the terminal with an image ID of 1, but we got a response
	/// back that indicated it was for an image with an ID of 2. This could theoretically happen if
	/// we are sending multiple images to the terminal quickly one after another and the terminal
	/// responses end up getting interleaved. There are a few ways around this:
	/// 1. When displaying an image, lock stdout until you get a response from the terminal. This
	///    ensures that you can't have two images sent at the same time in such a way that their
	///    responses get interleaved.
	/// 2. Don't use [`ImageId`]s, but instead use [`ImageNumbers`] - that way if a transmission
	///    returns an error due to this interleaving, we can easily retry without having to delete
	///    an image first.
	#[error(
		"The ID of type {ty:?} seen in the terminal response was different from the one we sent (was {found}, expected {expected})"
	)]
	DifferentIdInResponse {
		/// The type of the identifier whose returned value was different from what we expected
		ty: IdentifierType,
		/// The value returned by the terminal, still held by a string. This could be a parseable
		/// u32 that just didn't equal what we expected, but could also just be any string
		found: String,
		/// The value that we expected to get for this type
		expected: AnyValueOrSpecific<NonZeroU32>
	},
	/// If we got an ID in the response, but we didn't expect to get such an ID. This could also
	/// be caused by the same thing as [`Self::DifferentIdInResponse`].
	#[error(
		"The terminal response provided a id of type {ty:?}, but we didn't sent one as part of the request (with a value of '{value}')"
	)]
	IdInResponseButNotInRequest {
		/// The type of this identifier which we did not expect
		ty: IdentifierType,
		/// The value accompanying this identifier
		value: String
	},
	/// When we expected to get an ID in the response, but didn't see such an ID.
	#[error(
		"We expected to receive an ID of type {ty:?} with value {val} in the response, but saw no id of that type"
	)]
	MissingId {
		/// The type of the identifier which we expected to see
		ty: IdentifierType,
		/// The value we expected to see - If you pass in an [`crate::NumberOrId::Number`] when
		/// transmitting, we expect to see an [`crate::ImageId`] returned, but don't care what its value
		/// is, so this will be [`AnyValueOrSpecific::Any`]. In other instances, it will be
		/// [`AnyValueOrSpecific::Specific`]
		val: AnyValueOrSpecific<NonZeroU32>
	},
	/// In the response, we got an option which we didn't expect and thus don't know how to handle.
	/// The value contained inside is the key of this unexpected option.
	#[error("Unknown terminal response key '{0}'")]
	UnknownResponseKey(String),
	/// The terminal reported an error in its response, but it wasn't formatted like
	/// `<code>:<reason>` (where code is something like `EINVAL`), so we don't know how to handle
	/// it.
	#[error(
		"The terminal returned an error, but it was unparseable; was '{0}' but expected <code>:<reason>"
	)]
	MalformedError(String),
	/// The terminal reported an error, but it wasn't any code that we recognized so we can't
	/// process it into a structured enum
	#[error(
		"The terminal resported an error with an unknown error code '{code}' and accompanying reason '{reason}'"
	)]
	UnknownErrorCode {
		/// The code that they sent - normally something like `EINVAL` (but not `EINVAL` because we
		/// handle that correctly)
		code: String,
		/// The reason that the terminal provided to accompany this code and give it more context
		reason: String
	}
}

/// Errors that the underlying terminal could return to us in reponse to us writing to the terminal
/// (e.g. with [`Action::write_transmit_to`]). The exact list of error codes is not
/// specified by the specification, so this is non-exhaustive
///
/// [`Action::write_transmit_to`]: crate::action::Action::write_transmit_to
#[non_exhaustive]
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum TerminalError {
	/// "ENOENT" error code - no such entity
	#[error("No such entity was found: {0}")]
	NoEntity(String),
	/// "EINVAL" error code - invalid argument
	#[error("Invalid argument: {0}")]
	InvalidArgument(String),
	/// "EBADF" error code - Bad file, such as when you provide a path to a file that doesn't exist
	#[error("Bad file: {0}")]
	BadFile(String),
	/// "ENODATA" error code - No Data, such as when no data is sent over the [`Direct`] medium
	///
	/// [`Direct`]: crate::medium::Medium::Direct
	#[error("No Data: {0}")]
	NoData(String),
	/// "EFBIG" error code - too much data sent to the terminal for the provided
	/// [`crate::ImageDimensions`]
	#[error("File Too Large: {0}")]
	FileTooLarge(String)
}

/// An error that can occur when we try to decode a [`TerminalError`] from an error code and
/// reason (both in the form of a [`&str`])
pub struct UnknownErrorCode<'a>(pub(crate) &'a str);

impl<'a> TryFrom<(&'a str, &str)> for TerminalError {
	type Error = UnknownErrorCode<'a>;
	fn try_from(value: (&'a str, &str)) -> Result<Self, Self::Error> {
		let s = value.1.to_owned();
		Ok(match value.0 {
			"ENOENT" => Self::NoEntity(s),
			"EINVAL" => Self::InvalidArgument(s),
			"EBADF" => Self::BadFile(s),
			"ENODATA" => Self::NoData(s),
			"EFBIG" => Self::FileTooLarge(s),
			x => return Err(UnknownErrorCode(x))
		})
	}
}

/// An error that can arise when transmitting an image to the terminal. This sort of error can also
/// occur when just sending an image to query about it.
#[derive(thiserror::Error, Debug)]
pub enum TransmitError<InputError: Error> {
	/// The writer returned an error when we tried to write to it
	#[error("Couldn't write to writer: {0}")]
	Writing(std::io::Error),
	/// An error occurred when we tried to read a response from the terminal after transmitting the
	/// image
	#[error("Couldn't read input after transmitting: {0}")]
	ReadingInput(InputError),
	/// We got a response from the terminal, but the data it contained didn't match the format we
	/// expected and we don't know exactly what to do with it.
	#[error("Couldn't parse response from the terminal: {0}")]
	ParsingResponse(ParseError),
	/// We got a response from the terminal and parsed it correctly. It contained an error, such as
	/// that we sent nonsense data or it can't find the file specified or something
	#[error("The backing terminal returned an error: {0}")]
	Terminal(TerminalError)
}

impl<E: PartialEq + Error> PartialEq for TransmitError<E> {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(Self::Writing(e1), Self::Writing(e2)) => e1.to_string() == e2.to_string(),
			(Self::ReadingInput(e1), Self::ReadingInput(e2)) => e1 == e2,
			(Self::ParsingResponse(e1), Self::ParsingResponse(e2)) => e1 == e2,
			(Self::Terminal(e1), Self::Terminal(e2)) => e1 == e2,
			// We are adding this exhaustive match here for the first half of the wildcard to
			// ensure this fn is updated if we add more variants
			(
				Self::Writing(_)
				| Self::ReadingInput(_)
				| Self::ParsingResponse(_)
				| Self::Terminal(_),
				_
			) => false
		}
	}
}
