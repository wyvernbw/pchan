//! Struct for facilitating Tmux passthrough

use std::io::Write;

/// A struct to facilitate writing to tmux running on top of a terminal. In most cases, `W` is
/// going to be [`std::io::Stdout`].
///
/// This struct is made specifically to be used with this library. It cannot be arbitrarily used to
/// escape data to pass through Tmux.
pub struct TmuxWriter<W: Write> {
	/// the inner writer
	inner: W,
	/// Whether we've written the initial `\x1bPtmux;` that is required at the beginning of our
	/// message. This should be reset to `false` after we detect the final `\\` written to the
	/// buffer.
	wrote_first: bool
}

impl<W: Write> TmuxWriter<W> {
	/// Create a new instance of [`Self`], wrapping the given struct.
	pub fn new(inner: W) -> Self {
		Self {
			inner,
			wrote_first: false
		}
	}
}

impl<W: Write> Write for TmuxWriter<W> {
	fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
		if !self.wrote_first {
			self.inner.write_all(b"\x1bPtmux;")?;
			self.wrote_first = true;
		}

		let mut last_x1b = 0;
		for found_idx in memchr::memmem::find_iter(buf, b"\x1b") {
			// So we want to copy over the slice, from and including right after the last idx
			// written, up to but not including, the byte that is 0x1b. This means that if
			// `last_x1b` is not 0, then it's the location of an x1b, and will correctly come right
			// after the x1b that we just inserted. So we are doubling correctly and it's all good.
			self.inner.write_all(&buf[last_x1b..found_idx])?;
			// then we write a x1b
			self.inner.write_all(&[0x1b])?;
			// and then save for next time
			last_x1b = found_idx;
		}

		self.inner.write_all(&buf[last_x1b..])?;

		Ok(buf.len())
	}

	fn flush(&mut self) -> std::io::Result<()> {
		self.inner.write_all(b"\x1b\\")?;
		self.wrote_first = false;
		self.inner.flush()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn only_encode_that_we_know_of() {
		let mut writer = TmuxWriter::new(Vec::new());
		writer
			.write_all(b"\x1b]1337;SetProfile=NewProfileName\x07")
			.unwrap();
		writer.flush().unwrap();
		let resulting = String::from_utf8(writer.inner).unwrap();
		assert_eq!(
			resulting,
			"\x1bPtmux;\x1b\x1b]1337;SetProfile=NewProfileName\x07\x1b\\"
		);
	}
}
