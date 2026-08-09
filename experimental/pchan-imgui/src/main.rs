mod winit_glue;

use crate::winit_glue::{AppWindow, FrameLimit};
use imgui::{Condition, MouseCursor, TextureId, WindowFlags};
use imgui_wgpu::{Texture, TextureConfig};
use imgui_winit_support::WinitPlatform;
use miette::{Context, IntoDiagnostic};
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::run::Runner;
use pchan_executor::block_on;
use pchan_utils::{InitTracingArgs, default};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event_loop::ActiveEventLoop;

fn main() -> miette::Result<()> {
    pchan_utils::init_tracing(InitTracingArgs {
        panic_hook: false,
        file:       true,
        stdout:     false,
    });
    let mut pchan = PchanImgui::new()?;
    pchan.run();
    Ok(())
}

struct PchanImgui {
    gpu:            Arc<pchan_gpu::Renderer>,
    emu:            Emu,
    runner:         Runner,
    imgui_renderer: imgui_wgpu::Renderer,
    imgui:          imgui::Context,
    platform:       WinitPlatform,
    last_frame:     Instant,
    window:         Option<AppWindow>,
    last_cursor:    Option<MouseCursor>,

    dp_tex_id: TextureId,

    running: bool,
}

impl PchanImgui {
    fn new() -> miette::Result<Self> {
        // Setup
        let mut imgui = imgui::Context::create();
        let platform = imgui_winit_support::WinitPlatform::new(&mut imgui);
        let renderer_config = imgui_wgpu::RendererConfig {
            texture_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            ..imgui_wgpu::RendererConfig::new()
        };
        let mut gpu = block_on(pchan_gpu::Renderer::new());
        let mut renderer =
            imgui_wgpu::Renderer::new(&mut imgui, &gpu.device, &gpu.queue, renderer_config);
        let mut emu = Emu::new();
        let bios_path = std::env::var("PCHAN_BIOS")
            .into_diagnostic()
            .wrap_err("PCHAN_BIOS env var not set")?
            .parse::<PathBuf>()
            .wrap_err("invalid bios path")?;
        emu.set_bios_path(bios_path);
        emu.load_bios().into_diagnostic()?;
        emu.cpu.jump_to_bios();
        gpu.connect_emu(&mut emu);
        let gpu = Arc::new(gpu);
        gpu.clone().start();
        let runner = Runner::new().with_config(pchan_emu::run::RunnerConfig {
            force_mode: Some(pchan_emu::run::RunnerMode::Dynarec),
        });

        let mut audio_task = AudioTask::new()?;
        pchan_bind::bind_audio(&mut audio_task, &mut emu);
        let audio_stream = audio_task.start()?;
        std::mem::forget(audio_stream);

        // allocate texture for emulator display
        let dp_tex_config = TextureConfig {
            size: wgpu::Extent3d {
                width: 320,
                height: 240,
                ..Default::default()
            },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            sampler_desc: wgpu::wgt::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..default()
            },
            ..default()
        };
        let texture = Texture::new(&gpu.device, &renderer, dp_tex_config);
        let dp_tex_id = renderer.textures.insert(texture);

        Ok(PchanImgui {
            gpu,
            imgui_renderer: renderer,
            imgui,
            window: None,
            platform,
            last_frame: Instant::now(),
            last_cursor: None,
            dp_tex_id,
            emu,
            running: false,
            runner,
        })
    }

    fn replace_display_texture(&mut self, size: (u32, u32)) {
        // allocate texture for emulator display
        let dp_tex_config = TextureConfig {
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                ..Default::default()
            },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            sampler_desc: wgpu::wgt::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..default()
            },
            ..default()
        };
        let texture = Texture::new(&self.gpu.device, &self.imgui_renderer, dp_tex_config);
        let dp_tex_id = self
            .imgui_renderer
            .textures
            .replace(self.dp_tex_id, texture);
    }

    fn render_pchan(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(view) = self.imgui_renderer.textures.get(self.dp_tex_id) else {
            tracing::warn!("no texture view!");
            return;
        };
        let view = view.view();
        // let mut encoder = self
        //     .gpu
        //     .device
        //     .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label:                    None,
            color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });
        self.gpu.draw_display(&mut rpass);
    }

    fn ui(&mut self, ev: &ActiveEventLoop, dt: Duration) {
        let fb_size = self.window.as_ref().map(|w| {
            let s = w.window.inner_size();
            [
                s.width as f32 / w.hidpi_factor as f32,
                s.height as f32 / w.hidpi_factor as f32,
            ]
        });
        let Some(fb_size) = fb_size else {
            return;
        };

        let ui = self.imgui.new_frame();

        let window = ui.window("emu-ctrl");
        let old_running = self.running;
        window
            .size([320.0, 240.0], Condition::FirstUseEver)
            .position([400.0, 200.0], Condition::FirstUseEver)
            .build(|| {
                ui.text(format!("Frametime: {dt:?}"));
                ui.checkbox("running", &mut self.running);
            });

        let window = ui.window("Emulator Output");
        let mut new_tex_size = [0.0; 2];
        window
            // .size([320.0, 240.0], Condition::FirstUseEver)
            .size(fb_size, Condition::Always)
            // .position([400.0, 200.0], Condition::FirstUseEver)
            .position([0.0, 0.0], Condition::Always)
            .flags(
                WindowFlags::NO_TITLE_BAR
                    // | WindowFlags::NO_RESIZE
                    | WindowFlags::NO_MOVE
                    | WindowFlags::NO_BACKGROUND
                    | WindowFlags::NO_INPUTS,
            )
            .build(|| {
                let size = ui.content_region_avail();
                new_tex_size = size;

                let mut display_uniforms = self.gpu.display_uniforms.lock().unwrap();
                display_uniforms.screen_rect =
                    pchan_gpu::glam::U16Vec2::new(size[0] as u16, size[1] as u16);

                imgui::Image::new(self.dp_tex_id, size).build(ui);
            });

        if self.last_cursor != ui.mouse_cursor() {
            self.last_cursor = ui.mouse_cursor();
            let app_window = self.window.as_mut().unwrap();
            self.platform.prepare_render(ui, &app_window.window);
        }

        self.update_run_loop(ev, old_running);
        if let Some(tex) = self.imgui_renderer.textures.get(self.dp_tex_id)
            && (tex.width() != new_tex_size[0] as u32 || tex.height() != new_tex_size[1] as u32)
        {
            self.replace_display_texture((new_tex_size[0] as u32, new_tex_size[1] as u32));
        }
    }

    /// called every frame
    fn update(&mut self, dt: Duration, encoder: &mut wgpu::CommandEncoder) {
        if !self.running {
            return;
        }
        while !self.emu.consume_vblank_signal() {
            self.runner.execute(&mut self.emu);
        }
        self.render_pchan(encoder);
    }

    fn update_run_loop(&mut self, _ev: &ActiveEventLoop, old_running: bool) {
        match (old_running, self.running) {
            (false, true) => {
                println!("set game loop");
            }
            (true, false) => {
                println!("set event loop");
            }
            _ => {}
        }
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color::BLACK
    }
}
