use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gpui::prelude::*;
use gpui::{AppContext, Render, *};
use gpui_component::button::*;
use gpui_component::text::markdown;
use gpui_component::{ActiveTheme, Root, Theme, ThemeConfig};
use kanal::{Receiver, Sender};
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_utils::setup_tracing;

actions!(app, [Quit]);

fn main() -> miette::Result<()> {
    setup_tracing();

    gpui_platform::application()
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(move |cx| {
            gpui_component::init(cx);
            let theme = Theme::global_mut(cx);
            theme.apply_config(&Rc::new(ThemeConfig {
                mode: gpui_component::ThemeMode::Dark,
                ..Default::default()
            }));
            theme.background = theme.background.opacity(0.8);

            cx.set_window_appearance(Some(WindowAppearance::VibrantDark));

            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.on_action(|_: &Quit, cx| {
                cx.quit();
            });

            _ = cx.open_window(
                gpui::WindowOptions {
                    window_background: WindowBackgroundAppearance::Blurred,
                    ..Default::default()
                },
                |win, cx| {
                    let view = Debugger::new(win, cx).unwrap();
                    let view = cx.new(|_| view);
                    let theme = cx.theme().clone();
                    cx.new(|cx| Root::new(view, win, cx).bg(theme.background))
                },
            );
            cx.activate(true);
        });

    Ok(())
}

struct Debugger {
    emu:         Emu,
    runner:      Runner,
    running:     bool,
    renderer:    Arc<pchan_gpu::Renderer>,
    last_render: Instant,

    target:       wgpu::Texture,
    target_buf:   wgpu::Buffer,
    display_tx:   Sender<SurfaceState>,
    display_task: JoinHandle<()>,
}

use miette::IntoDiagnostic;
use pchan_emu::run::Runner;
use pchan_gpu::wgpu;

impl Debugger {
    pub fn new(window: &Window, cx: &App) -> miette::Result<Self> {
        let mut emu = Emu::new();
        emu.set_bios_path(std::env::var("PCHAN_BIOS").into_diagnostic()?);
        emu.load_bios().into_diagnostic()?;
        emu.cpu.jump_to_bios();
        emu.tty.set_tracing();

        let mut audio_task = AudioTask::new()?;
        pchan_bind::bind_audio(&mut audio_task, &mut emu);
        let audio_stream = audio_task.start()?;
        std::mem::forget(audio_stream);

        let gpu = pchan_gpu::Renderer::try_new();
        let gpu = cx.foreground_executor().block_on(gpu).into_diagnostic()?;

        let mut dp = gpu.display_uniforms.lock().unwrap();
        dp.screen_rect.x = 320;
        dp.screen_rect.y = 240;
        let (target, target_buf) = create_target(&gpu, &mut dp);
        drop(dp);

        gpu.connect_emu(&mut emu);
        let gpu = Arc::new(gpu);
        gpu.clone().start();

        let (display_tx, display_rx) = kanal::unbounded::<SurfaceState>();

        let display_task = std::thread::spawn({
            let gpu = gpu.clone();
            move || {
                while let Ok(surface) = display_rx.recv() {
                    draw_display(
                        &gpu,
                        &surface.target,
                        &surface.target_buf,
                        &surface.fifo_tx,
                        &surface.fifo_rx,
                        surface.buffer_mapped,
                    );
                }
            }
        });

        Ok(Self {
            emu,
            renderer: gpu,
            running: true,
            runner: Runner::new().with_config(pchan_emu::run::RunnerConfig {
                force_mode: Some(pchan_emu::run::RunnerMode::Interpreter),
            }),
            last_render: Instant::now(),

            target,
            target_buf,
            display_tx,
            display_task,
        })
    }
}

fn create_target(
    gpu: &pchan_gpu::Renderer,
    dp: &mut MutexGuard<'_, pchan_gpu::DisplayUniforms>,
) -> (wgpu::Texture, wgpu::Buffer) {
    let width = dp.screen_rect.x * 4;
    let width = (width + 255) & !255; // align to 256
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label:           None,
        size:            wgpu::Extent3d {
            width:                 dp.screen_rect.x as u32,
            height:                dp.screen_rect.y as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Bgra8UnormSrgb,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats:    &[wgpu::TextureFormat::Bgra8UnormSrgb],
    });
    let target_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:              None,
        size:               width as u64 * dp.screen_rect.y as u64,
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (target, target_buf)
}

impl Render for Debugger {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::prelude::Context<Self>,
    ) -> impl gpui::prelude::IntoElement {
        let surface_state = window.use_state(cx, |_, _| {
            let (fifo_tx, fifo_rx) = kanal::bounded(2);
            SurfaceState {
                target: self.target.clone(),
                target_buf: self.target_buf.clone(),
                fifo_tx,
                fifo_rx,
                buffer_mapped: Arc::new(AtomicBool::new(false)),
            }
        });
        let surface = self.pchan_game_surface(surface_state);
        let theme = cx.theme();
        let frame_time = self.last_render.elapsed();

        if self.running {
            self.display_tx
                .send(surface.state.read(cx).clone())
                .unwrap();
            while !self.emu.consume_vblank_signal() {
                self.runner.execute(&mut self.emu);
            }
            window.request_animation_frame();
        }

        self.last_render = Instant::now();

        div()
            .h_full()
            .flex_col()
            .child(header(cx, &frame_time, self.display_tx.len()))
            .child(
                div()
                    .h_full()
                    .flex_grow_1()
                    .child(surface.relative().w_full().h_full())
                    .child(
                        div()
                            .id("main")
                            .text_color(theme.foreground)
                            .p_4()
                            .flex_col()
                            .overflow_scroll()
                            .child(markdown("Hello world!").selectable(true))
                            .child(Button::new("Wow!").child("Wow!")),
                    ),
            )
    }
}

fn header<T>(cx: &Context<T>, frame_time: &Duration, frames_in_flight: usize) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_sm()
        .bg(theme.title_bar)
        .px_4()
        .py_1()
        .flex()
        .gap_4()
        .border_color(theme.title_bar_border)
        .border_b_1()
        .text_color(theme.table_head_foreground)
        .child("🐷🎗️ P-ちゃん")
        .child(div().flex_grow_1())
        .child(
            markdown(format!("frame: {:02}ms", frame_time.as_millis()))
                .font_family(&theme.mono_font_family),
        )
        .child(
            markdown(format!("frames in flight: {}", frames_in_flight))
                .font_family(&theme.mono_font_family),
        )
}

struct RawSurface {
    div:      Div,
    renderer: Arc<pchan_gpu::Renderer>,
    state:    Entity<SurfaceState>,
    vram:     Box<[u16]>,
}

#[derive(Clone)]
struct SurfaceState {
    target:        wgpu::Texture,
    target_buf:    wgpu::Buffer,
    fifo_tx:       Sender<Arc<RenderImage>>,
    fifo_rx:       Receiver<Arc<RenderImage>>,
    buffer_mapped: Arc<AtomicBool>,
}

impl IntoElement for RawSurface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for RawSurface {
    type RequestLayoutState = DivFrameState;
    type PrepaintState = Option<Hitbox>;

    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let state = self.state.read(cx);
        // let img = draw_display(
        //     &self.renderer,
        //     &state.target,
        //     &state.target_buf,
        //     &state.fifo_tx,
        //     &state.fifo_rx,
        //     state.buffer_mapped.clone(),
        // );
        let img = get_display_result(&self.renderer, &state.fifo_rx);
        // let mut buf = vec![0u8; 1024 * 512 * 4];
        // for (i, pixel) in self.vram.iter().enumerate() {
        //     let pixel = Rgb5::new_with_raw_value(*pixel);
        //     buf[i * 4 + 0] = ((pixel.b().value() as u16) * 256 / 32) as u8;
        //     buf[i * 4 + 1] = ((pixel.g().value() as u16) * 256 / 32) as u8;
        //     buf[i * 4 + 2] = ((pixel.r().value() as u16) * 256 / 32) as u8;
        //     buf[i * 4 + 3] = 255;
        // }
        // let frame = image::Frame::new(image::ImageBuffer::from_raw(1024, 512, buf).unwrap());
        window
            .paint_image(
                bounds,
                Bounds::new(
                    Point {
                        x: 0.0.into(),
                        y: 0.0.into(),
                    },
                    Size {
                        width:  state.target.width().into(),
                        height: state.target.height().into(),
                    },
                ),
                Corners::default(),
                img,
                0,
                false,
            )
            .unwrap();
    }

    fn id(&self) -> Option<ElementId> {
        Element::id(&self.div)
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        Element::source_location(&self.div)
    }

    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        Element::request_layout(&mut self.div, id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let width = bounds.size.width;
        let height = bounds.size.height;
        let width = width.as_f32() as u16;
        let height = height.as_f32() as u16;
        let mut dp = self.renderer.display_uniforms.lock().unwrap();

        if width != dp.screen_rect.x || height != dp.screen_rect.y {
            dp.screen_rect.x = width;
            dp.screen_rect.y = height;
            let (target, target_buf) = create_target(&self.renderer, &mut dp);
            self.state.update(cx, |state, _| {
                state.target = target;
                state.target_buf = target_buf;
            })
        }

        Element::prepaint(
            &mut self.div,
            id,
            inspector_id,
            bounds,
            request_layout,
            window,
            cx,
        )
    }
}

impl Styled for RawSurface {
    #[doc = " Returns a reference to the style memory of this element."]
    fn style(&mut self) -> &mut StyleRefinement {
        self.div.style()
    }
}

impl Debugger {
    pub fn pchan_game_surface(&mut self, state: Entity<SurfaceState>) -> RawSurface {
        // let _ = self.emu.gpu.lock_vram();
        RawSurface {
            div: div(),
            renderer: self.renderer.clone(),
            state,
            vram: Box::new([]),
        }
    }
}

pub(crate) fn draw_display(
    gpu: &pchan_gpu::Renderer,
    target: &wgpu::Texture,
    target_buf: &wgpu::Buffer,
    fifo_tx: &Sender<Arc<RenderImage>>,
    fifo_rx: &Receiver<Arc<RenderImage>>,
    buffer_mapped: Arc<AtomicBool>,
) {
    use pchan_gpu::wgpu;
    use wgpu::*;

    let target_view = target.create_view(&TextureViewDescriptor {
        label:             None,
        format:            Some(TextureFormat::Bgra8UnormSrgb),
        dimension:         None,
        aspect:            wgpu::TextureAspect::All,
        base_mip_level:    0,
        mip_level_count:   None,
        base_array_layer:  0,
        array_layer_count: None,
        usage:             Some(TextureUsages::RENDER_ATTACHMENT),
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
            view:           &target_view,
            resolve_target: None,
            ops:            wgpu::Operations {
                load:  wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice:    None,
        })],
        depth_stencil_attachment: None,
        label:                    Some("display render pass"),
        timestamp_writes:         None,
        occlusion_query_set:      None,
        multiview_mask:           None,
    });
    gpu.draw_display(&mut rpass);
    drop(rpass);
    let unpadded = target.width() * 4;
    let padded = (unpadded + 255) & !255; // align to 256
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: target_buf,
            layout: TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(padded),
                rows_per_image: Some(target.height()),
            },
        },
        target.size(),
    );

    let commands = encoder.finish();

    gpu.queue.submit([commands]);

    target_buf.map_async(MapMode::Read, .., {
        let fifo_tx = fifo_tx.clone();
        let target_shadow = target_buf.clone();
        let width = target.width();
        let height = target.height();
        let buffer_mapped = buffer_mapped.clone();

        move |_| {
            buffer_mapped.store(true, std::sync::atomic::Ordering::Release);
            let data = target_shadow.get_mapped_range(..).unwrap();
            let mut buf = Vec::with_capacity(data.len());
            for row in data.chunks(padded as usize) {
                buf.extend_from_slice(&row[..unpadded as usize]);
            }

            let frame =
                image::Frame::new(image::ImageBuffer::from_raw(width, height, buf).unwrap());
            let img = RenderImage::new([frame]);
            drop(data);
            target_shadow.unmap();

            buffer_mapped.store(false, std::sync::atomic::Ordering::Release);
            let img = Arc::new(img);
            fifo_tx.send(img).unwrap();
        }
    });
}

fn get_display_result(
    gpu: &pchan_gpu::Renderer,
    fifo_rx: &Receiver<Arc<RenderImage>>,
) -> Arc<RenderImage> {
    use wgpu::*;

    loop {
        match fifo_rx.try_recv_realtime() {
            Ok(Some(frame)) => {
                return frame;
            }
            _ => {
                _ = gpu.device.poll(PollType::Poll);
            }
        }
    }
}
