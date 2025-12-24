#![feature(proc_macro_hygiene)]
#![feature(try_blocks)]

pub mod window {
    use std::sync::Arc;

    use thiserror::Error;
    use tracing::instrument;
    use winit::{
        application::ApplicationHandler,
        error::OsError,
        event::WindowEvent,
        event_loop::ActiveEventLoop,
        window::{Window, WindowAttributes, WindowId},
    };

    use crate::gpu::GpuState;

    pub struct EmuDisplay {
        state: Option<GpuState>,
    }

    impl ApplicationHandler<GpuState> for EmuDisplay {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            match self.handle_resume(event_loop) {
                Ok(_) => {}
                Err(ResumeErr::OnCreateWindow(err)) => panic!("could not create window: {err}"),
                Err(ResumeErr::OnCreateGpuState(err)) => {
                    panic!("could not create gpu state: {err}")
                }
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, event: GpuState) {
            event.window.request_redraw();
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            let Some(state) = &mut self.state else {
                return;
            };
            match event {
                // TODO: handle resize
                WindowEvent::Resized(new_size) => {
                    tracing::info!(?new_size);
                }
                WindowEvent::RedrawRequested => {
                    match state.render() {
                        Ok(_) => {}
                        // Reconfigure the surface if it's lost or outdated
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = state.window.inner_size();
                            // TODO: resize correctly
                            // state.resize(size.width, size.height);
                        }
                        Err(e) => {
                            tracing::error!("{e}");
                        }
                    }
                }
                WindowEvent::CloseRequested => {}
                _ => {}
            }
        }
    }

    #[derive(Error, Debug)]
    enum ResumeErr {
        #[error(transparent)]
        OnCreateWindow(#[from] OsError),
        #[error(transparent)]
        OnCreateGpuState(color_eyre::Report),
    }

    impl EmuDisplay {
        #[instrument(ret, err, skip_all)]
        fn handle_resume(&mut self, event_loop: &ActiveEventLoop) -> Result<(), ResumeErr> {
            let window =
                event_loop.create_window(WindowAttributes::default().with_title("P-chan!"))?;
            let win = Arc::new(window);

            let state =
                smol::block_on(GpuState::try_new(win)).map_err(ResumeErr::OnCreateGpuState)?;

            self.state = Some(state);
            Ok(())
        }

        pub const fn new() -> Self {
            EmuDisplay { state: None }
        }
    }

    impl Default for EmuDisplay {
        fn default() -> Self {
            Self::new()
        }
    }
}

pub mod gpu {
    use std::sync::Arc;

    use bon::Builder;
    use color_eyre::eyre::eyre;
    use tracing::instrument;
    use wgpu::{
        DeviceDescriptor, RenderPassColorAttachment, RenderPassDescriptor, SurfaceConfiguration,
        TextureViewDescriptor,
    };
    use winit::window::Window;

    #[derive(Debug, Builder)]
    pub struct GpuState {
        pub(crate) surface: wgpu::Surface<'static>,
        pub(crate) device: wgpu::Device,
        pub(crate) queue: wgpu::Queue,
        pub(crate) config: wgpu::SurfaceConfiguration,
        pub(crate) is_surface_configured: bool,
        pub(crate) window: Arc<Window>,
    }

    impl GpuState {
        #[instrument(err, skip_all)]
        pub(crate) async fn try_new(window: Arc<Window>) -> color_eyre::Result<GpuState> {
            let size = window.inner_size();

            // The instance is a handle to our GPU
            // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });

            let surface = instance.create_surface(window.clone()).unwrap();

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference:       wgpu::PowerPreference::default(),
                    compatible_surface:     Some(&surface),
                    force_fallback_adapter: false,
                })
                .await?;

            let (device, queue) = adapter
                .request_device(&DeviceDescriptor {
                    ..Default::default()
                })
                .await?;

            let surface_config = SurfaceConfiguration {
                ..surface
                    .get_default_config(&adapter, size.width, size.height)
                    .ok_or(eyre!("failed to create surface default config"))?
            };

            surface.configure(&device, &surface_config);

            Ok(Self {
                surface,
                device,
                queue,
                config: surface_config,
                is_surface_configured: true,
                window,
            })
        }

        pub(crate) fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
            self.window.request_redraw();

            if !self.is_surface_configured {
                return Ok(());
            }

            let output = self.surface.get_current_texture()?;
            let view = output
                .texture
                .create_view(&TextureViewDescriptor::default());

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

            {
                let _render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
            }

            // submit will accept anything that implements IntoIter
            self.queue.submit(std::iter::once(encoder.finish()));
            output.present();

            Ok(())
        }
    }
}
