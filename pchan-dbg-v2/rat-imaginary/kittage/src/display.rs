//! Structs and functions useful for controlling exactly where an image is displayed in the
//! terminal surface and what happens when it is displayed

use std::io::Write;

use crate::{
	CURSOR_MOVEMENT_POLICY_KEY, DISPLAY_COLS_KEY, DISPLAY_HEIGHT_SLICE_KEY, DISPLAY_ROWS_KEY,
	DISPLAY_START_X_KEY, DISPLAY_START_Y_KEY, DISPLAY_WIDTH_SLICE_KEY, ImageId, PARENT_ID_KEY,
	PARENT_PLACEMENT_KEY, PIXEL_X_OFFSET_KEY, PIXEL_Y_OFFSET_KEY, RELATIVE_HORIZ_CELL_OFFSET_KEY,
	RELATIVE_VERT_CELL_OFFSET_KEY, VIRTUAL_PLACEMENT_KEY, WriteUint as _, Z_INDEX_KEY
};

/// How the cursor should act after an image is displayed
#[derive(Default, PartialEq, Clone, Copy, Debug)]
#[repr(u8)]
pub enum CursorMovementPolicy {
	/// The cursor should move to the cell directly following the last cell of the image
	#[default]
	MoveToAfterImage = 0,
	/// The cursor should not move at all
	DontMove = 1
}

/// The details of how a specific image should be dislpayed onto the terminal surface. used in e.g.
/// [`Action::TransmitAndDisplay`] and [`Action::Display`]
///
/// [`Action::TransmitAndDisplay`]: crate::action::Action::TransmitAndDisplay
/// [`Action::Display`]: crate::action::Action::Display
#[derive(Default, Debug, PartialEq, Clone)]
pub struct DisplayConfig {
	/// The location to display it at
	pub location: DisplayLocation,
	/// Cursor movement policy; see enum documentation for more details
	pub cursor_movement: CursorMovementPolicy,
	/// If true, creates a virtual placement for a unicode placeholder
	pub create_virtual_placement: bool,
	/// The id of a parent image for relative placement
	pub parent_id: Option<ImageId>,
	/// The id of a placement in the parent image for relative placement
	pub parent_placement: Option<ImageId>
}

impl DisplayConfig {
	/// Encode this [`DisplayConfig`] into a writer to send to kitty
	pub(crate) fn write_to<W: Write>(&self, mut writer: W) -> std::io::Result<W> {
		writer = self.location.write_to(writer)?;
		if self.cursor_movement != CursorMovementPolicy::default() {
			write!(
				writer,
				",{CURSOR_MOVEMENT_POLICY_KEY}={}",
				self.cursor_movement as u8
			)?;
		}
		if self.create_virtual_placement {
			write!(writer, ",{VIRTUAL_PLACEMENT_KEY}=1")?;
		}

		if let Some(parent_id) = self.parent_id {
			writer = writer.write_uint::<PARENT_ID_KEY, _>(parent_id.get())?;
		}

		if let Some(parent_placement) = self.parent_placement {
			writer = writer.write_uint::<PARENT_PLACEMENT_KEY, _>(parent_placement.get())?;
		}

		Ok(writer)
	}
}

/// Fields to specify exactly where to place an image (and specifically what part of an image) on
/// the terminal surface. Used within [`DisplayConfig`]
#[derive(Default, Debug, PartialEq, Clone)]
pub struct DisplayLocation {
	/// The left edge (in pixels) of the image area to display. If > 0, this can be used to crop
	/// the image from the left side
	pub x: u32,
	/// The top edge (in pixels) of the image area to display. If > 0, this can be used to crop the
	/// image from the top
	pub y: u32,
	/// The width (in pixels) of the image area to display. If left unspecified, the entire width
	/// is used.
	pub width: u32,
	/// The height (in pixels) of the image area to display. If left unspecified, the entire width
	/// is used.
	pub height: u32,
	/// The x-offset within the first cell at which to start displaying the image
	pub x_offset: usize,
	/// The y-offset within the first cell at which to start displaying the image
	pub y_offset: usize,
	/// The number of columns to display the image over
	pub columns: u16,
	/// The number of rows to display the image over
	pub rows: u16,
	/// The z-index vertical stacking order ofthe image
	pub z_index: i32,
	/// The offset cells in the horizontal direction for relative placement
	pub horizontal_offset: i32,
	/// The offset in cells in the vertical direction for relative placement
	pub vertical_offset: i32
}

impl DisplayLocation {
	/// Encode this [`DisplayLocation`] into a writer to send to kitty
	fn write_to<W: Write>(&self, w: W) -> std::io::Result<W> {
		w.write_uint::<DISPLAY_START_X_KEY, _>(self.x)?
			.write_uint::<DISPLAY_START_Y_KEY, _>(self.y)?
			.write_uint::<DISPLAY_WIDTH_SLICE_KEY, _>(self.width)?
			.write_uint::<DISPLAY_HEIGHT_SLICE_KEY, _>(self.height)?
			.write_uint::<PIXEL_X_OFFSET_KEY, _>(self.x_offset)?
			.write_uint::<PIXEL_Y_OFFSET_KEY, _>(self.y_offset)?
			.write_uint::<DISPLAY_COLS_KEY, _>(self.columns)?
			.write_uint::<DISPLAY_ROWS_KEY, _>(self.rows)?
			.write_uint::<Z_INDEX_KEY, _>(self.z_index)?
			.write_uint::<RELATIVE_HORIZ_CELL_OFFSET_KEY, _>(self.horizontal_offset)?
			.write_uint::<RELATIVE_VERT_CELL_OFFSET_KEY, _>(self.vertical_offset)
	}
}
