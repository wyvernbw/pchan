extern crate alloc;

use alloc::sync::Arc;
use core::alloc::Allocator;
use core::mem;
use fltk::button::Button;
use fltk::frame::Frame;
use fltk::group::Flex;
use miette::IntoDiagnostic;
use pchan_audio::AudioTask;
use pchan_emu::Emu;

use fltk::app::{self, App};
use fltk::prelude::*;
use fltk::window::Window;

fn main() {
    let app = App::default()
        .with_scheme(fltk::app::Scheme::Plastic)
        .load_system_fonts();
    let mut win = Window::new(0, 0, 640, 480, "Pーちゃん").center_screen();
    win.set_frame(fltk::enums::FrameType::NoBox);
    win.make_resizable(true);
    let mut frame = Flex::default_fill().column();
    frame.end();

    let (s, r) = app::channel::<Msg>();
    let (tx, rx) = kanal::unbounded_async::<Msg>();

    Button::new(0, 0, 32, 32, "button")
        .with_label("click")
        .center_of(&win)
        .emit(s, Msg::Heartbeat);
    win.end();
    win.show();

    while app.wait() {
        if let Some(msg) = r.recv() {
            println!("{msg:?}");
            let frame1 = Frame::default().with_label("hiii");
            frame.add(&frame1);
            frame.fixed(&frame1, 24);
            frame.layout();
            frame.redraw();
        }
    }
}

#[derive(Debug, Clone)]
enum Msg {
    Heartbeat,
}

fn create_emu<A: Allocator + Copy>(alloc: A) -> miette::Result<Emu<A>> {
    let mut emu = Emu::new_in(alloc);
    emu.set_bios_path(std::env::var("PCHAN_BIOS").into_diagnostic()?);
    emu.load_bios(alloc).into_diagnostic()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();

    let mut audio_task = AudioTask::new()?;
    pchan_bind::bind_audio(&mut audio_task, &mut emu);
    let audio_stream = audio_task.start()?;
    mem::forget(audio_stream);

    let gpu = pchan_gpu::Renderer::try_new();
    let gpu = smol::block_on(gpu).into_diagnostic()?;

    let mut dp = gpu.display_uniforms.app.lock().unwrap();
    dp.screen_rect.x = 320;
    dp.screen_rect.y = 240;
    drop(dp);

    gpu.connect_emu(&mut emu);
    let gpu = Arc::new(gpu);
    gpu.clone().start();

    Ok(emu)
}
