use std::sync::Arc;
use std::time::{Duration, Instant};

use imgui::FontSource;
use wgpu::SurfaceTexture;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes};

use crate::PchanImgui;

pub(crate) struct AppWindow {
    pub(crate) window:        Arc<Window>,
    pub(crate) surface_desc:  wgpu::SurfaceConfiguration,
    pub(crate) surface:       wgpu::Surface<'static>,
    pub(crate) surface_dirty: bool,
    pub(crate) hidpi_factor:  f64,
    pub(crate) frame_limit:   FrameLimit,
}

pub(crate) enum FrameLimit {
    Vsync,
    Manual(Duration),
    Unlimited,
}

impl PchanImgui {
    pub(crate) fn run(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Wait);
        event_loop.run_app(self).unwrap();
    }
    pub(crate) fn set_frame_limiter(&mut self, limit: FrameLimit) {
        let Some(window) = &mut self.window else {
            return;
        };
        window.frame_limit = limit;
        window.frame_limit.configure(&mut window.surface_desc);
        window.surface_dirty = true;
    }
    pub(crate) fn configure_surface(&mut self) {
        let Some(window) = &mut self.window else {
            return;
        };
        if window.surface_dirty {
            window
                .surface
                .configure(&self.gpu.device, &window.surface_desc);
            window.surface_dirty = false;
        }
    }
    pub(crate) fn present(&mut self, frame: SurfaceTexture, dt: Duration) {
        let Some(window) = &mut self.window else {
            return;
        };
        self.gpu.queue.present(frame);
        window.frame_limit.block(dt);
        if self.running {
            window.window.request_redraw();
        }
    }
}

impl FrameLimit {
    pub(crate) fn configure(&self, surface_desc: &mut wgpu::SurfaceConfiguration) {
        match self {
            FrameLimit::Vsync => {
                surface_desc.present_mode = wgpu::PresentMode::AutoVsync;
            }
            FrameLimit::Unlimited | FrameLimit::Manual(_) => {
                surface_desc.present_mode = wgpu::PresentMode::AutoNoVsync
            }
        }
    }
    pub(crate) fn block(&self, elapsed: Duration) {
        match self {
            FrameLimit::Vsync | FrameLimit::Unlimited => {}
            FrameLimit::Manual(duration) => {
                spin_sleep::sleep(duration.saturating_sub(elapsed));
            }
        }
    }
}

impl winit::application::ApplicationHandler for PchanImgui {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );
        let size = window.inner_size();
        let hidpi_factor = window.scale_factor();
        let surface = self.gpu.instance.create_surface(window.clone()).unwrap();
        let mut platform = imgui_winit_support::WinitPlatform::new(&mut self.imgui);
        platform.attach_window(
            self.imgui.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );

        let surface_desc = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:                        wgpu::TextureFormat::Bgra8UnormSrgb,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode:                    wgpu::CompositeAlphaMode::Auto,
            view_formats:                  vec![wgpu::TextureFormat::Bgra8Unorm],
            color_space:                   wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&self.gpu.device, &surface_desc);
        self.window = Some(AppWindow {
            window,
            surface_desc,
            surface,
            hidpi_factor,
            frame_limit: FrameLimit::Vsync,
            surface_dirty: false,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        self.configure_surface();
        if let Some(window) = &mut self.window {
            match event {
                WindowEvent::RedrawRequested => {}
                _ => {
                    window.window.request_redraw();
                }
            }
        }
        match &event {
            WindowEvent::Resized(size) => {
                let window = self.window.as_mut().unwrap();
                window.surface_desc.width = size.width;
                window.surface_desc.height = size.height;
                window
                    .surface
                    .configure(&self.gpu.device, &window.surface_desc);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let window = self.window.as_mut().unwrap();
                window.hidpi_factor = *scale_factor;
                let font_size = (13.0 * window.hidpi_factor) as f32;
                self.imgui.fonts().clear();
                self.imgui.fonts().add_font(&[FontSource::DefaultFontData {
                    config: Some(imgui::FontConfig {
                        oversample_h: 1,
                        pixel_snap_h: true,
                        size_pixels: font_size,
                        ..Default::default()
                    }),
                }]);
                self.imgui_renderer.reload_font_texture(
                    &mut self.imgui,
                    &self.gpu.device,
                    &self.gpu.queue,
                );
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let Key::Named(NamedKey::Escape) = event.logical_key
                    && event.state.is_pressed()
                {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                let dt = frame_start.duration_since(self.last_frame);
                self.last_frame = frame_start;

                self.imgui.io_mut().update_delta_time(dt);

                let window = self.window.as_mut().unwrap();
                let frame = match window.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame) => frame,
                    // Suboptimal is fine to render with — likely an
                    // upcoming resize will reconfigure the surface.
                    wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    wgpu::CurrentSurfaceTexture::Timeout
                    | wgpu::CurrentSurfaceTexture::Occluded => return,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        window
                            .surface
                            .configure(&self.gpu.device, &window.surface_desc);
                        return;
                    }
                    other => {
                        eprintln!("get_current_texture error: {other:?}");
                        return;
                    }
                };
                self.platform
                    .prepare_frame(self.imgui.io_mut(), &window.window)
                    .expect("Failed to prepare frame");

                let mut encoder: wgpu::CommandEncoder = self
                    .gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // run emulator frame (if running)
                self.update(dt, &mut encoder);
                // create ui
                self.ui(event_loop, dt);

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:                    None,
                    color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                        view:           &view,
                        resolve_target: None,
                        ops:            wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(self.clear_color()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice:    None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes:         None,
                    occlusion_query_set:      None,
                    multiview_mask:           None,
                });

                self.imgui_renderer
                    .render(
                        self.imgui.render(),
                        &self.gpu.queue,
                        &self.gpu.device,
                        &mut rpass,
                    )
                    .expect("Rendering failed");

                drop(rpass);

                let window = self.window.as_mut().unwrap();
                self.gpu.queue.submit(Some(encoder.finish()));
                window.window.pre_present_notify();

                self.present(frame, dt);
            }
            _ => (),
        }

        let window = self.window.as_mut().unwrap();
        self.platform.handle_event::<()>(
            self.imgui.io_mut(),
            &window.window,
            &Event::WindowEvent { window_id, event },
        );
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: ()) {
        let window = self.window.as_mut().unwrap();
        self.platform.handle_event::<()>(
            self.imgui.io_mut(),
            &window.window,
            &Event::UserEvent(event),
        );
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let window = self.window.as_mut().unwrap();
        self.platform.handle_event::<()>(
            self.imgui.io_mut(),
            &window.window,
            &Event::DeviceEvent { device_id, event },
        );
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.running
            && let Some(window) = self.window.as_ref()
        {
            window.window.request_redraw();
        }
    }
}
