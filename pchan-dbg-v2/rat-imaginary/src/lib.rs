pub use kittage::ImageDimensions;
pub use kittage::PixelFormat;
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufRead, BufReader, Stdin, Stdout, Write},
    num::NonZero,
    str::Utf8Error,
    time::Instant,
};

use kittage::{
    InputReader,
    action::Action,
    delete::{ClearOrDelete, WhichToDelete},
    image::{Image, ImageFromShmFailureStep},
    medium::ShmError,
    tmux::TmuxWriter,
};
use ratatui::widgets::StatefulWidget;
use thiserror::Error;

struct InternalInputReader {
    stdin:      BufReader<Stdin>,
    temp:       Vec<u8>,
    seen_first: bool,
}

#[derive(Error, Debug)]
enum InternalReadErr {
    #[error("terminal response is not valid utf8: {}", .0)]
    ResponseNotUtf8(#[from] Utf8Error),
    #[error(transparent)]
    IoErr(#[from] std::io::Error),
}

impl InputReader for InternalInputReader {
    type Error = InternalReadErr;

    fn read_esc_delimited_str(&mut self, buf: &mut String) -> Result<(), Self::Error> {
        self.temp.clear();
        // let mut esc_count = 0;
        // loop {
        //     let mut readbuf = [0u8; 1024];
        //     match self.stdin.read(&mut readbuf) {
        //         Err(err) => return Err(err.into()),
        //         Ok(len) => {
        //             self.temp.extend_from_slice(&readbuf[..len]);
        //             while let Some(esc_idx) = self.temp.iter().position(|b| *b == 0x1b) {
        //                 esc_count += 1;
        //                 match esc_count {
        //                     1 => self.temp = self.temp.split_off(esc_idx + 1),
        //                     2 => {
        //                         let str = str::from_utf8(&self.temp[..esc_idx]);
        //                         buf.clear();
        //                         buf.extend(str);
        //                         return Ok(());
        //                     }
        //                     _ => unreachable!(),
        //                 }
        //             }
        //         }
        //     }
        // }
        match self.stdin.read_until(0x1b, &mut self.temp) {
            Ok(len) => {
                if self.seen_first {
                    self.temp.truncate(len.saturating_sub(1));
                    let str = str::from_utf8(&self.temp);
                    buf.clear();
                    buf.extend(str);
                    Ok(())
                } else {
                    self.seen_first = true;
                    self.read_esc_delimited_str(buf)
                }
            }
            Err(_) => todo!(),
        }
    }
}

impl InternalInputReader {
    fn new() -> Self {
        Self {
            stdin:      BufReader::new(std::io::stdin()),
            temp:       vec![],
            seen_first: false,
        }
    }
}

pub struct ImageState {
    current_hash: Option<u64>,
    queue:        [Option<Frame>; 2],
    writer:       Option<ImageWriter>,
    idx:          usize,
}

enum ImageWriter {
    Plain(Stdout),
    Tmux(TmuxWriter<Stdout>),
}

impl ImageWriter {
    fn new() -> Self {
        let is_tmux = std::env::var("TERM")
            .map(|value| value.starts_with("tmux"))
            .unwrap_or(false);
        match is_tmux {
            true => ImageWriter::Tmux(TmuxWriter::new(std::io::stdout())),
            false => ImageWriter::Plain(std::io::stdout()),
        }
    }
}

impl Write for ImageWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            ImageWriter::Plain(stdout) => stdout.write(buf),
            ImageWriter::Tmux(tmux_writer) => tmux_writer.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            ImageWriter::Plain(stdout) => stdout.flush(),
            ImageWriter::Tmux(tmux_writer) => tmux_writer.flush(),
        }
    }
}

struct Frame {
    id: NonZero<u32>,
}

impl ImageState {
    pub fn new() -> Self {
        Self {
            current_hash: None,
            queue:        [const { None }; 2],
            idx:          0,
            writer:       Some(ImageWriter::new()),
        }
    }

    pub fn write_image(
        &mut self,
        (format, data): (PixelFormat, Vec<u8>),
    ) -> Result<(), ImageFromShmFailureStep<ShmError>> {
        let seed = Instant::now();
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let seed = hasher.finish();
        let img = Image::shm_from((format, data.to_vec()), &format!("{}-{seed}", self.idx))?;
        let writer = self.writer.take();

        if let Some(writer) = writer {
            let Ok((writer, res)) =
                Action::Transmit(img).execute(writer, InternalInputReader::new())
            else {
                self.writer = Some(ImageWriter::new());
                self.clear_backbuffer();
                self.swap_buffers();
                self.clear_backbuffer();
                self.swap_buffers();
                return Ok(());
            };

            self.writer = Some(writer);
            if let Some(id) = res {
                self.queue[self.idx] = Some(Frame { id });
            }
        }

        Ok(())
    }

    pub fn swap_buffers(&mut self) {
        self.idx = (self.idx + 1) % 2;
    }

    fn backbuffer_idx(&self) -> usize {
        1 - self.idx
    }

    fn take_writer(&mut self) -> ImageWriter {
        self.writer.take().expect("writer must be initialized")
    }

    fn clear_buffer(&mut self, idx: usize) {
        if let Some(frame) = &self.queue[idx] {
            let action = Action::Delete(kittage::delete::DeleteConfig {
                effect: ClearOrDelete::Delete,
                which:  WhichToDelete::ImageId(frame.id, None),
            });
            let writer = self.take_writer();
            let (writer, _) = action
                .execute(writer, InternalInputReader::new())
                .expect("failed to clear buffer");
            self.writer = Some(writer);
        }
    }

    pub fn clear_backbuffer(&mut self) {
        self.clear_buffer(self.backbuffer_idx());
    }

    pub fn clear_frontbuffer(&mut self) {
        self.clear_buffer(self.idx);
    }
}

impl Default for ImageState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ImageWidget;

impl ImageWidget {
    pub fn render(
        self,
        area: ratatui::prelude::Rect,
        frame: &mut ratatui::Frame,
        state: &mut ImageState,
    ) {
        frame.set_cursor_position((area.x, area.y));
        StatefulWidget::render(self, area, frame.buffer_mut(), state);
    }
}

impl StatefulWidget for ImageWidget {
    type State = ImageState;

    fn render(
        self,
        area: ratatui::prelude::Rect,
        buf: &mut ratatui::prelude::Buffer,
        state: &mut Self::State,
    ) {
        if let Some(frame) = &state.queue[state.idx] {
            let stdout = state.writer.take().expect("must have writer");
            let Ok((mut stdout, id)) = Action::Display {
                image_id:     frame.id,
                placement_id: NonZero::new(1).unwrap(),
                config:       kittage::display::DisplayConfig {
                    location:                 kittage::display::DisplayLocation {
                        x:                 0,
                        y:                 0,
                        width:             640,
                        height:            480,
                        x_offset:          0,
                        y_offset:          0,
                        columns:           area.width,
                        rows:              area.height,
                        z_index:           1,
                        horizontal_offset: 0,
                        vertical_offset:   0,
                    },
                    cursor_movement:          kittage::display::CursorMovementPolicy::DontMove,
                    create_virtual_placement: false,
                    parent_id:                None,
                    parent_placement:         None,
                },
            }
            .execute(stdout, InternalInputReader::new()) else {
                state.writer = Some(ImageWriter::new());
                state.clear_backbuffer();
                state.swap_buffers();
                state.clear_backbuffer();
                state.swap_buffers();
                return;
            };

            // if let Some(id) = id {
            //     let [id_extra, id_r, id_g, id_b] = id.get().to_be_bytes();
            //     let id_color = format!("\x1b[38;2;{id_r};{id_g};{id_b}m");
            //     for y in area.y..(area.y + area.height) {
            //         write!(stdout, "\x1b[{};{}H", y + 1, area.x + 1).unwrap();
            //         write!(stdout, "\x1b[{id_color}").unwrap();
            //         for x in area.x..(area.x + area.width) {
            //             if let Some(cell) = buf.cell_mut((x, y)) {
            //                 write!(
            //                     stdout,
            //                     "\u{10EEEE}{row_d}{col_d}{id_extra}",
            //                     row_d = DIACRITICS[(y - area.y) as usize],
            //                     col_d = DIACRITICS[(x - area.x) as usize],
            //                     id_extra = DIACRITICS[id_extra as usize]
            //                 )
            //                 .unwrap();
            //                 cell.set_skip(true);
            //             }
            //         }
            //         write!(stdout, "\x1b[39m\n").unwrap();
            //     }
            //     stdout.flush();
            // }
            state.writer = Some(stdout);

            state.swap_buffers();
        }
    }
}

impl Drop for ImageState {
    fn drop(&mut self) {
        self.clear_backbuffer();
        self.swap_buffers();
        self.clear_backbuffer();
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crossterm::event::{KeyCode, KeyEvent};
    use image::{DynamicImage, GenericImage, Rgba};
    use ratatui::layout::Position;

    use super::*;
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let i = (h * 6.0).floor() as u32;
        let f = h * 6.0 - i as f32;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);

        let (r, g, b) = match i % 6 {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };

        ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    #[test]
    fn it_works() {
        ratatui::run(|term| {
            let mut img_state = ImageState::new();
            let start = Instant::now();
            let mut x_dir = 2;
            let mut y_dir = 1;
            let mut pos = Position::new(1, 1);
            let mut counter = 0;
            loop {
                term.draw(|frame| {
                    frame.render_widget(counter.to_string(), frame.area());
                    let border = frame.area();
                    let t = start.elapsed().as_secs_f32();
                    let mut img = DynamicImage::new_rgb8(640, 480);
                    for y in 0..480 {
                        for x in 0..640 {
                            let hue = (x as f32 / 640.0 + t * 0.4) % 1.0;

                            let wave = f32::sin(x as f32 * 0.01) * 0.1;
                            let hue = (hue + wave + y as f32 / 480.0 * 0.3) % 1.0;

                            let (r, g, b) = hsv_to_rgb(hue, 0.5, 1.0);
                            img.put_pixel(x, y, Rgba([r, g, b, 255]));
                        }
                    }
                    img_state
                        .write_image(Image::fmt_and_data_from(img))
                        .unwrap();

                    pos = pos.offset(ratatui::layout::Offset { x: x_dir, y: y_dir });
                    if pos.x == 0 || pos.x >= border.width.saturating_sub(64) {
                        x_dir *= -1;
                    }
                    if pos.y == 0 || pos.y >= border.height.saturating_sub(48 / 2) {
                        y_dir *= -1;
                    }

                    ImageWidget.render(
                        ratatui::layout::Rect {
                            x:      pos.x,
                            y:      pos.y,
                            width:  64,
                            height: 24,
                        },
                        frame,
                        &mut img_state,
                    );
                })
                .unwrap();

                if let Ok(true) = crossterm::event::poll(Duration::from_millis(16)) {
                    match crossterm::event::read().unwrap().as_key_event() {
                        None => {}
                        Some(KeyEvent {
                            code: KeyCode::Char('q'),
                            ..
                        }) => break,
                        _ => counter += 1,
                    }
                }
            }
        });
    }
}
