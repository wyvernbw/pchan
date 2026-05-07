//! Types and functions to facilitate [`crate::action::Action::Delete`]

use std::{io::Write, ops::RangeInclusive};

use crate::{ImageId, ImageNumber, PlacementId};

/// What to delete when using [`crate::action::Action::Delete`]
#[derive(PartialEq, Debug, Clone)]
pub struct DeleteConfig {
	/// Whether to just clear the specified images from the screen ([`ClearOrDelete::Clear`]) or
	/// completely delete them from memory ([`ClearOrDelete::Delete`])
	pub effect: ClearOrDelete,
	/// Specifying exactly how to determine which images to delete
	pub which: WhichToDelete
}

/// Whether to clear the specified images from the screen or completely delete them from memory
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum ClearOrDelete {
	/// just clear them from the screen
	Clear,
	/// clear them from the screen AND delete them from memory
	Delete
}

impl ClearOrDelete {
	/// Make `c` uppercase if `self == Self::Delete`, otherwise keep it the same. This corresponds
	/// with the behavior that kitty expects us to use to specify what to do with the deletions.
	fn maybe_upper(self, c: char) -> char {
		match self {
			Self::Delete => c.to_ascii_uppercase(),
			Self::Clear => c
		}
	}
}

/// An (x, y) index of a specific cell on the terminal (i.e. specific location where a single
/// character could be printed)
#[derive(PartialEq, Debug, Clone)]
pub struct CellLocation {
	/// The x-index
	pub x: u16,
	/// The y-index
	pub y: u16
}

/// A [`CellLocation`] with an associated z-index (such as is used with
/// [`crate::display::DisplayLocation::z_index`])
#[derive(PartialEq, Debug, Clone)]
pub struct CellLocationZ {
	/// The x and y of the location
	pub x_y: CellLocation,
	/// The z-index
	pub z: i32
}

/// A filter to specify exactly which images should be deleted
#[derive(PartialEq, Debug, Clone)]
pub enum WhichToDelete {
	/// Every single visible image (does not include non-visible)
	All,
	/// Only delete those with the specific [`ImageId`]. If a [`PlacementId`] is specified, then
	/// this only deletes images with both the [`ImageId`] AND the [`PlacementId`]
	ImageId(ImageId, Option<PlacementId>),
	/// Only delete those with the specific [`ImageNumber`]. If a [`PlacementId`] is specified, then
	/// this only deletes images with both the [`ImageId`] AND the [`PlacementId`]
	NewestWithNumber(ImageNumber, Option<PlacementId>),
	/// Only delete images which intersect with the cursor's current position
	IntersectingWithCursor,
	/// Delete all animation frames
	AnimationFrames,
	/// Delete all images that intersect at all with the given cell
	PlacementsIntersectingCell(CellLocation),
	/// Delete all images that intersect at all with the given cell (whose location includes a
	/// z-index that also must be matched)
	PlacementsIntersectingCellWithZ(CellLocationZ),
	/// Delete all images whose ids are contained in the given range
	IdRange(RangeInclusive<ImageId>),
	/// Delete all placements that intersect with the given cell column
	PlacementsIntersectingColumn(u16),
	/// Delete all placements that intersect with the given cell row
	PlacementsIntersectingRow(u16),
	/// Delete all placements that intersect with the given z-index
	PlacementsWithZIndex(i32)
}

impl DeleteConfig {
	/// Encode this [`DeleteConfig`] into a string to be sent to kitty to communicate this deletion
	pub(crate) fn write_to<W: Write>(&self, mut w: W) -> std::io::Result<W> {
		let e = self.effect;

		write!(w, ",d=")?;

		match &self.which {
			WhichToDelete::All => write!(w, "{}", e.maybe_upper('a'))?,
			WhichToDelete::ImageId(img_id, placement_id) => {
				write!(w, "{},i={img_id}", e.maybe_upper('i'))?;
				if let Some(p_id) = placement_id {
					write!(w, ",p={p_id}")?;
				}
			}
			WhichToDelete::NewestWithNumber(img_num, placement_id) => {
				write!(w, "{},I={img_num}", e.maybe_upper('n'))?;
				if let Some(p_id) = placement_id {
					write!(w, ",p={p_id}")?;
				}
			}
			WhichToDelete::IntersectingWithCursor => write!(w, "{}", e.maybe_upper('c'))?,
			WhichToDelete::AnimationFrames => write!(w, "{}", e.maybe_upper('f'))?,
			WhichToDelete::PlacementsIntersectingCell(CellLocation { x, y }) =>
				write!(w, "{},x={x},y={y}", e.maybe_upper('p'))?,
			WhichToDelete::PlacementsIntersectingCellWithZ(CellLocationZ {
				x_y: CellLocation { x, y },
				z
			}) => write!(w, "{},x={x},y={y},z={z}", e.maybe_upper('q'))?,
			WhichToDelete::IdRange(range) => write!(
				w,
				"{},x={},y={}",
				e.maybe_upper('r'),
				range.start(),
				range.end()
			)?,
			WhichToDelete::PlacementsIntersectingColumn(col) =>
				write!(w, "{},x={col}", e.maybe_upper('x'))?,
			WhichToDelete::PlacementsIntersectingRow(row) =>
				write!(w, "{},y={row}", e.maybe_upper('y'))?,
			WhichToDelete::PlacementsWithZIndex(z) => write!(w, "{},z={z}", e.maybe_upper('z'))?
		}

		write!(w, "\x1b\\")?;
		Ok(w)
	}
}
