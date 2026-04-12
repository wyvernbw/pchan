#![feature(try_blocks)]

use color_eyre::{
    Result,
    eyre::{Context, bail},
};
use egui_winit::winit::{
    self,
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop, EventLoopProxy},
    window::WindowAttributes,
};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use egui::{Ui, mutex::RwLock};
use egui_wgpu::{
    self, CallbackTrait, ScreenDescriptor,
    wgpu::{
        self, Surface, TextureUsages, TextureView, TextureViewDescriptor,
        wgt::CommandEncoderDescriptor,
    },
};
use pchan_emu::{Emu, bootloader::Bootloader, dynarec_v2::PipelineV2, io::vblank::VBlank};
use wgpu::{
    BufferUsages,
    util::{BufferInitDescriptor, DeviceExt},
};

fn main() -> Result<()> {
    color_eyre::install()?;
    pchan_utils::init_tracing()
        .panic_hook(false)
        .indicatif(false)
        .call();
    Main::run()?;
    Ok(())
}

struct PchanDbgEgui {
    pchan_rd:   Arc<pchan_gpu::Renderer>,
    pchan_emu:  Arc<RwLock<Emu>>,
    egui_state: egui_winit::State,
    egui_ctx:   egui::Context,
    egui_rd:    egui_wgpu::Renderer,
    window:     Arc<winit::window::Window>,
    surface:    Surface<'static>,
}

impl PchanDbgEgui {
    fn new(window: winit::window::Window, events: EventLoopProxy<UserEvent>) -> Result<Self> {
        let window = Arc::new(window);
        let size = window.inner_size();
        let mut pchan_emu = Emu::default();
        let bios_path = std::env::var("PCHAN_BIOS")
            .wrap_err("PCHAN_BIOS env var not set.")?
            .parse::<PathBuf>()
            .wrap_err("PCHAN_BIOS var contains invalid path.")?;
        pchan_emu.set_bios_path(bios_path);
        pchan_emu.load_bios()?;
        pchan_emu.cpu.jump_to_bios();
        pchan_emu.tty.set_tracing();

        let mut pchan_rd = smol::block_on(pchan_gpu::Renderer::new());
        pchan_rd.connect_emu(&mut pchan_emu);
        let pchan_rd = Arc::new(pchan_rd);
        let pchan_emu = Arc::new(RwLock::new(pchan_emu));

        {
            let pchan_emu = pchan_emu.clone();
            pchan_rd.clone().start();
            std::thread::spawn(move || {
                let pchan_emu = pchan_emu;
                let mut pipe = PipelineV2::new(&pchan_emu.read());
                loop {
                    pipe = pipe.run_once(&mut pchan_emu.write()).unwrap();
                    if pchan_emu.write().consume_vblank_signal() {
                        events.send_event(UserEvent::RenderFinished).unwrap();
                        let target = Duration::from_nanos(16_666_667); // 59.94hz, more precise
                        let deadline = pchan_emu.read().gpu.last_vblank + target;
                        let now = Instant::now();

                        if let Some(remaining) = deadline.checked_duration_since(now) {
                            // coarse sleep
                            let coarse = remaining.saturating_sub(Duration::from_micros(200));
                            if !coarse.is_zero() {
                                std::thread::sleep(coarse);
                            }
                            // spin tail
                            while Instant::now() < deadline {}
                        }

                        pchan_emu.write().gpu.last_vblank = Instant::now();
                    }
                }
            });
        }

        let surface = pchan_rd.instance.create_surface(window.clone())?;

        let egui_ctx = egui::Context::default();
        let id = egui_ctx.viewport_id();
        let egui_state = egui_winit::State::new(egui_ctx.clone(), id, &window, None, None, None);

        let egui_rd = egui_wgpu::Renderer::new(
            &pchan_rd.device,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            egui_wgpu::RendererOptions {
                msaa_samples:                  1,
                depth_stencil_format:          None,
                dithering:                     false,
                predictable_texture_filtering: false,
            },
        );

        let pchan_dbg_egui = Self {
            surface,
            egui_ctx,
            pchan_rd,
            pchan_emu,
            egui_state,
            window,
            egui_rd,
        };
        pchan_dbg_egui.configure_surface();
        Ok(pchan_dbg_egui)
    }

    fn configure_surface(&self) {
        let size = self.window.inner_size();
        let surface_caps = self.surface.get_capabilities(&self.pchan_rd.adapter);
        // tracing::info!("surface capabilities = {surface_caps:#?}");
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:                        surface_format,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  surface_caps.present_modes[0],
            alpha_mode:                    surface_caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        self.surface.configure(&self.pchan_rd.device, &config);
    }

    pub fn draw(
        &mut self,
        screen_descriptor: ScreenDescriptor,
        mut run_ui: impl FnMut(&mut Ui),
    ) -> Result<()> {
        // self.state.set_pixels_per_point(window.scale_factor() as f32);
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run_ui(raw_input, |ui| {
            run_ui(ui);
        });

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_rd.update_texture(
                &self.pchan_rd.device,
                &self.pchan_rd.queue,
                *id,
                image_delta,
            );
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.configure_surface();
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            value => bail!("cannot get surface texture: {value:?}"),
        };
        let window_surface_view = output.texture.create_view(&TextureViewDescriptor {
            label:             None,
            format:            None,
            dimension:         None,
            aspect:            wgpu::TextureAspect::All,
            base_mip_level:    0,
            mip_level_count:   None,
            base_array_layer:  0,
            array_layer_count: None,
            usage:             Some(TextureUsages::RENDER_ATTACHMENT),
        });
        let mut encoder = self
            .pchan_rd
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        self.egui_rd.update_buffers(
            &self.pchan_rd.device,
            &self.pchan_rd.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );
        let mut rpass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                    view:           &window_surface_view,
                    resolve_target: None,
                    ops:            wgpu::Operations {
                        load:  wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice:    None,
                })],
                depth_stencil_attachment: None,
                label:                    Some("egui main render pass"),
                timestamp_writes:         None,
                occlusion_query_set:      None,
                multiview_mask:           None,
            })
            .forget_lifetime();
        self.egui_rd.render(&mut rpass, &tris, &screen_descriptor);
        drop(rpass);
        self.pchan_rd
            .queue
            .submit(std::iter::once(encoder.finish()));
        output.present();
        for x in &full_output.textures_delta.free {
            self.egui_rd.free_texture(x)
        }

        Ok(())
    }

    fn custom_painting(ui: &mut egui::Ui, pchan_rd: Arc<pchan_gpu::Renderer>) {
        let (rect, response) =
            ui.allocate_exact_size(egui::Vec2::splat(300.0), egui::Sense::drag());

        let callback = egui_wgpu::Callback::new_paint_callback(rect, RenderCallback(pchan_rd));
        ui.painter().add(callback);
    }
}

struct RenderCallback(Arc<pchan_gpu::Renderer>);

impl CallbackTrait for RenderCallback {
    // fn prepare(
    //     &self,
    //     _device: &eframe::wgpu::Device,
    //     _queue: &eframe::wgpu::Queue,
    //     _screen_descriptor: &egui_wgpu::ScreenDescriptor,
    //     _egui_encoder: &mut eframe::wgpu::CommandEncoder,
    //     _callback_resources: &mut egui_wgpu::CallbackResources,
    // ) -> Vec<eframe::wgpu::CommandBuffer> {
    //     vec![]
    // }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut egui_wgpu::wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vertex_buf = &[0, 0, 0, 0, 0, 0];
        let vertex_buffer = self.0.device.create_buffer_init(&BufferInitDescriptor {
            label:    None,
            usage:    BufferUsages::VERTEX,
            contents: vertex_buf,
        });
        render_pass.set_pipeline(&self.0.display_pipeline);
        render_pass.set_bind_group(0, &self.0.display_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
    }
}

struct Main {
    app:   Option<PchanDbgEgui>,
    proxy: EventLoopProxy<UserEvent>,
}

impl Main {
    #[tracing::instrument(skip_all)]
    fn run() -> Result<()> {
        let event_loop = EventLoop::<UserEvent>::with_user_event()
            .build()
            .wrap_err("failed to create winit::EventLoop")?;
        event_loop.set_control_flow(ControlFlow::Wait);
        let proxy = event_loop.create_proxy();
        let mut main_app = Self { app: None, proxy };
        event_loop
            .run_app(&mut main_app)
            .wrap_err("error encountered while running pchan debugger")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    RenderFinished,
}

impl ApplicationHandler<UserEvent> for Main {
    #[tracing::instrument(skip_all)]
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }

        let res: Result<()> = try {
            let window = event_loop
                .create_window(WindowAttributes::default())
                .wrap_err("failed to create window")?;
            let pchan = PchanDbgEgui::new(window, self.proxy.clone())
                .wrap_err("failed to create pchan app")?;
            self.app = Some(pchan);
        };

        res.unwrap()
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        let Some(app) = &mut self.app else {
            return;
        };
        match event {
            UserEvent::RenderFinished => {
                tracing::info!("pushing rasterized frame");
                app.window.request_redraw()
            }
        }
    }

    #[tracing::instrument(skip_all)]
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(app) = &mut self.app else {
            return;
        };
        if window_id != app.window.id() {
            return;
        }
        match event {
            WindowEvent::Resized(_) => {
                app.configure_surface();
            }
            WindowEvent::RedrawRequested => {
                let pchan_rd = app.pchan_rd.clone();
                app.draw(
                    ScreenDescriptor {
                        size_in_pixels:   [
                            app.window.inner_size().width,
                            app.window.inner_size().height,
                        ],
                        pixels_per_point: app.window.scale_factor() as f32,
                    },
                    |ui| {
                        let pchan_rd = &pchan_rd;
                        ui.label("hello from egui!");
                        egui::Frame::canvas(ui.style()).show(ui, |ui| {
                            PchanDbgEgui::custom_painting(ui, pchan_rd.clone());
                        });
                    },
                )
                .unwrap();
            }
            _ => {
                let res = app.egui_state.on_window_event(&app.window, &event);
                if res.repaint {
                    app.window.request_redraw();
                }
            }
        }
    }
}
