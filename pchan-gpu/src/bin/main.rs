use winit::event_loop::EventLoop;

use pchan_gpu::window::EmuDisplay;

pub fn main() -> color_eyre::Result<()> {
    pchan_utils::setup_tracing();
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = EmuDisplay::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
