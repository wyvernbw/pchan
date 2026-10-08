#![allow(recursion_depth_exceeding_limit)]
#![feature(duration_millis_float)]

extern crate alloc;

use alloc::alloc::Global;
use alloc::sync::Arc;
use core::alloc::Allocator;
use core::mem;
use core::time::Duration;
use fltk::menu::{MenuFlag, SysMenuBar};
use miette::IntoDiagnostic;
use pchan_audio::AudioTask;
use pchan_bind::ringbuf::StaticRb;
use pchan_bind::ringbuf::traits::{Consumer, Observer, RingBuffer};
use pchan_emu::Emu;
use pchan_emu::run::{Runner, RunnerConfig, RunnerMode};
use pchan_gpu::wgpu::rwh::{HasDisplayHandle, HasWindowHandle};
use pchan_gpu::{Renderer, wgpu};
use pchan_steel::{SteelCtx, SteelExecutor};
use pchan_utils::{InitTracingArgs, default, init_tracing};
use std::time::Instant;

use fltk::app::{self, App};
use fltk::prelude::*;
use fltk::window::Window;

fn main() -> miette::Result<()> {
    init_tracing(InitTracingArgs {
        stdout:     false,
        file:       true,
        panic_hook: false,
    });
    let app = App::default()
        .with_scheme(fltk::app::Scheme::Plastic)
        .load_system_fonts();
    let mut win = Window::new(0, 0, 640, 480, "Pーちゃん").center_screen();

    let mut menu = SysMenuBar::default().with_size(400, 30);
    menu.add(
        "File/Open...\t",
        fltk::enums::Shortcut::Command | 'o',
        MenuFlag::Normal,
        |_| {
            println!("open");
        },
    );
    menu.add(
        "File/Quit\t",
        fltk::enums::Shortcut::Command | 'q',
        MenuFlag::Normal,
        |_| {
            app::quit();
        },
    );

    win.set_frame(fltk::enums::FrameType::NoBox);
    win.make_resizable(true);

    let (s, r) = app::channel::<Msg>();

    win.end();
    win.show();
    win.set_on_top();

    let (mut emu, gpu, mut surface) = smol::block_on(create_emu(Global, &win))?;
    let mut runner = Runner::new().with_config(RunnerConfig {
        force_mode: Some(RunnerMode::Dynarec),
        ..default()
    });
    let steel_ctx = SteelCtx::new();
    let steel_rx = steel_ctx.rx();
    let mut steel_exec = SteelExecutor::new();

    std::thread::spawn(move || {
        steel_ctx.repl().unwrap();
        s.send(Msg::Quit);
    });

    smol::spawn({
        let steel_rx = steel_rx.clone();
        async move {
            while let Ok(msg) = steel_rx.chan.1.recv().await {
                s.send(Msg::ReplCall(msg));
            }
        }
    })
    .detach();

    let mut frame_time_buf = StaticRb::<f32, 60>::default();

    s.send(Msg::Heartbeat {
        deadline: Instant::now(),
    });

    while app.wait() {
        let Some(msg) = r.recv() else { continue };
        match msg {
            Msg::Heartbeat { deadline } => {
                loop {
                    if Instant::now() > deadline {
                        break;
                    }
                    core::hint::spin_loop();
                }

                let elapsed = runner.run_until_vblank(&mut emu);
                if let Some(elapsed) = elapsed {
                    let elapsed_ms = elapsed.as_millis_f32();
                    let sleep_for = runner.sleep_time(elapsed);
                    frame_time_buf.push_overwrite(elapsed_ms);

                    draw_into(&mut surface, &gpu, &win);
                    steel_exec.handle_step(&mut emu, &mut runner);

                    smol::spawn(async move {
                        let now = Instant::now();
                        let spin_for = sleep_then_spin(sleep_for).await;
                        s.send(Msg::Heartbeat {
                            deadline: now + spin_for,
                        });
                    })
                    .detach();

                    if runner.frame_idx.is_multiple_of(60) {
                        let avg_elapsed = frame_time_buf.iter().copied().sum::<f32>()
                            / frame_time_buf.occupied_len() as f32;

                        win.set_label(&format!("Pーちゃん | {avg_elapsed:.2}ms"));
                    }
                }
            }
            Msg::Quit => break,
            Msg::ReplCall(call) => {
                steel_exec.handle_call(&mut emu, &mut runner, call);
                s.send(Msg::Heartbeat {
                    deadline: Instant::now(),
                });
            }
        }
    }

    Ok(())
}

async fn sleep_then_spin(duration: Duration) -> Duration {
    let sleep_for = duration.saturating_sub(Duration::from_millis(3));
    smol::Timer::after(sleep_for).await;

    duration.saturating_sub(sleep_for)
}

#[derive(Debug, Clone)]
enum Msg {
    Quit,
    Heartbeat { deadline: Instant },
    ReplCall(pchan_steel::Call),
}

async fn create_emu<A: Allocator + Copy>(
    alloc: A,
    win: &Window,
) -> miette::Result<(Emu<A>, Arc<Renderer>, SurfaceCtx)> {
    use wgpu::*;

    let mut emu = Emu::new_in(alloc);
    emu.set_bios_path(std::env::var("PCHAN_BIOS").into_diagnostic()?);
    emu.load_bios(alloc).into_diagnostic()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();

    let mut audio_task = AudioTask::new()?;
    pchan_bind::bind_audio(&mut audio_task, &mut emu);
    let audio_stream = audio_task.start()?;
    mem::forget(audio_stream);

    let instance = wgpu::Instance::default();
    let surface = surface_from_window(win, &instance);
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface:     Some(&surface),
            apply_limit_buckets:    true,
        })
        .await
        .into_diagnostic()?;
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor::default())
        .await
        .into_diagnostic()?;
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats[0];
    let surface_config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        color_space: SurfaceColorSpace::Auto,
        width: win.pixel_w() as _,
        height: win.pixel_h() as _,
        present_mode: PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![format],
    };
    surface.configure(&device, &surface_config);

    let gpu = Renderer::from_wgpu(instance, adapter, device, queue, format).into_diagnostic()?;

    gpu.connect_emu(&mut emu);
    let mut dp = gpu.display_uniforms.app.lock().unwrap();
    dp.screen_rect.x = 320;
    dp.screen_rect.y = 240;
    drop(dp);

    let gpu = Arc::new(gpu);
    gpu.clone().start();

    Ok((
        emu,
        gpu,
        SurfaceCtx {
            surface,
            surface_config,
        },
    ))
}

struct SurfaceCtx {
    surface:        wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}
fn surface_from_window(win: &Window, inst: &wgpu::Instance) -> wgpu::Surface<'static> {
    use wgpu::*;

    unsafe {
        inst.create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(win.display_handle().unwrap().as_raw()),
            raw_window_handle:  win.window_handle().unwrap().as_raw(),
        })
        .unwrap()
    }
}

impl SurfaceCtx {
    pub fn configure(&mut self, device: &wgpu::Device) {
        self.surface.configure(device, &self.surface_config);
    }

    pub fn handle_resize(&mut self, gpu: &pchan_gpu::Renderer, win: &Window) {
        if win.pixel_w() as u32 != self.surface_config.width
            || win.pixel_h() as u32 != self.surface_config.height
        {
            self.surface_config.width = win.pixel_w() as u32;
            self.surface_config.height = win.pixel_h() as u32;
            self.configure(&gpu.device);
        }
    }
}

fn draw_into(surface: &mut SurfaceCtx, gpu: &Renderer, win: &Window) {
    use wgpu::*;
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    surface.handle_resize(gpu, win);
    let stex = match surface.surface.get_current_texture() {
        CurrentSurfaceTexture::Success(stex) => stex,
        CurrentSurfaceTexture::Suboptimal(stex) => {
            surface.configure(&gpu.device);
            stex
        }
        _ => {
            surface.configure(&gpu.device);
            return;
        }
    };
    let view = stex.texture.create_view(&TextureViewDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
        label:                    None,
        color_attachments:        &[Some(RenderPassColorAttachment {
            view:           &view,
            depth_slice:    None,
            resolve_target: None,
            ops:            Operations {
                load:  LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes:         None,
        occlusion_query_set:      None,
        multiview_mask:           None,
    });
    gpu.draw_display(&mut rpass);

    drop(rpass);
    gpu.queue.submit([encoder.finish()]);
    gpu.queue.present(stex);
}
