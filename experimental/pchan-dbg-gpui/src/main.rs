#![allow(recursion_depth_exceeding_limit)]

#[path = "game-surface.rs"]
pub mod game_surface;

use std::num::ParseIntError;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gpui::prelude::*;
use gpui::{AppContext, Render, *};
use gpui_component::input::{Input, InputState};
use gpui_component::tab::TabBar;
use gpui_component::text::markdown;
use gpui_component::{ActiveTheme, Root, StyledExt, Theme, ThemeConfig, h_flex, v_flex};
use kanal::Sender;
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::cpu::REG_STR;
use pchan_utils::{hex, setup_tracing};

actions!(app, [Quit]);

fn main() -> miette::Result<()> {
    setup_tracing();

    gpui_platform::application()
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(move |cx| {
            gpui_component::init(cx);
            let theme = Theme::global_mut(cx);
            theme.apply_config(&Rc::new(ThemeConfig {
                mono_font_family: Some("GeistMono Nerd Font".into()),
                mode: gpui_component::ThemeMode::Dark,
                ..Default::default()
            }));

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
                    cx.new(|cx| Root::new(view, win, cx).h_full().bg(theme.background))
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

    cached_reg_names:    [SharedString; 32],
    cpu_control_reg_tab: usize,

    target:        wgpu::Texture,
    target_buf:    wgpu::Buffer,
    display_tx:    Sender<SurfaceState>,
    _display_task: JoinHandle<()>,
}

use miette::IntoDiagnostic;
use pchan_emu::run::Runner;
use pchan_gpu::wgpu;

use crate::game_surface::{SurfaceState, create_target, draw_display};

impl Debugger {
    pub fn new(_window: &Window, cx: &App) -> miette::Result<Self> {
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

        let _display_task = std::thread::spawn({
            let gpu = gpu.clone();
            move || {
                while let Ok(surface) = display_rx.recv() {
                    draw_display(
                        &gpu,
                        &surface.target,
                        &surface.target_buf,
                        &surface.fifo_tx,
                        surface.buffer_mapped,
                    );
                }
            }
        });

        let cached_reg_names = core::array::from_fn(|reg| {
            let reg = match reg as u8 {
                0 => "0",
                pchan_emu::cpu::FP => "fp",
                other => REG_STR[other as usize],
            };
            format!("${}", reg).into()
        });

        Ok(Self {
            emu,
            renderer: gpu,
            running: true,
            runner: Runner::new().with_config(pchan_emu::run::RunnerConfig {
                force_mode: Some(pchan_emu::run::RunnerMode::Dynarec),
            }),
            last_render: Instant::now(),

            cached_reg_names,
            cpu_control_reg_tab: 0,

            target,
            target_buf,
            display_tx,
            _display_task,
        })
    }
}

impl Render for Debugger {
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
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
        let _theme = cx.theme();

        self.last_render = Instant::now();

        if self.running {
            self.display_tx
                .send(surface.state.read(cx).clone())
                .unwrap();
            while !self.emu.consume_vblank_signal() {
                self.runner.execute(&mut self.emu);
            }
            window.request_animation_frame();
        }

        let frame_time = self.last_render.elapsed();
        let window_height = window.viewport_size().height;

        v_flex()
            .h(window_height)
            .child(header(cx, &frame_time, self.display_tx.len()))
            .child(
                div()
                    .h_full()
                    .bg(transparent_white())
                    .child(surface.absolute().w_full().h_full())
                    .child(self.debugger_ui(window, cx).w_full().h_full()),
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

impl Debugger {
    fn debugger_ui(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + Styled {
        let theme = cx.theme();

        v_flex()
            .id("main")
            .w_full()
            .h_full()
            .relative()
            .text_color(theme.foreground)
            .p_4()
            .bg(transparent_white())
            .child(div().flex_grow_1())
            // bottom panel
            .child(
                div().flex().min_h_0().max_h_72().flex_grow_1().child(
                    div()
                        .w_full()
                        .h_full()
                        .flex()
                        .text_sm()
                        .border_2()
                        .bg(theme.background)
                        .border_color(theme.border)
                        .corner_radii(Corners::all(8.0.into()))
                        .p_2()
                        .child(self.cpu_controls(window, cx).h_full()),
                ),
            )
    }
}

fn parse_hex_register(str: &str) -> Result<u32, ParseIntError> {
    if str == "0x" {
        return Ok(0);
    }
    u32::from_str_radix(str.trim_prefix("0x"), 16)
}

impl Debugger {
    fn cpu_controls(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Debugger>,
    ) -> impl IntoElement + Styled {
        let theme = cx.theme().clone();

        let tabs = TabBar::new("segmented-tabs")
            .segmented()
            .selected_index(self.cpu_control_reg_tab)
            .cursor_pointer()
            .on_click(cx.listener(|view, index, _, cx| {
                view.cpu_control_reg_tab = *index;
                cx.notify();
            }))
            .children(vec!["CPU", "COP0", "GTE"]);

        let regs = self.emu.cpu.gpr.iter().copied().enumerate().map({
            |(r, value)| {
                let reg_id = &self.cached_reg_names[r];
                let input_state = window.use_keyed_state(reg_id.clone(), cx, |win, cx| {
                    let reg_value: SharedString = hex(value).to_string().into();
                    InputState::new(win, cx)
                        .default_value(reg_value)
                        .validate(|value, _| parse_hex_register(value).is_ok())
                });

                use gpui_component::input::InputEvent;

                cx.subscribe_in(
                    &input_state,
                    window,
                    move |view, input_state, event, win, cx| {
                        if let InputEvent::PressEnter { .. } | InputEvent::Blur = event {
                            match parse_hex_register(&input_state.read(cx).value()) {
                                Ok(reg_value) => view.emu.cpu.gpr[r] = reg_value,
                                // TODO: handle error
                                Err(_err) => {}
                            }
                            input_state.update(cx, |state, cx| {
                                let value: SharedString =
                                    hex(view.emu.cpu.gpr[r]).to_string().into();
                                state.set_value(value, win, cx);
                            })
                        }
                    },
                )
                .detach();

                let color = match value {
                    0 => theme.colors.muted_foreground,
                    _ => theme.colors.foreground,
                };

                div()
                    .flex()
                    .id(reg_id.clone())
                    .font_family(&cx.theme().mono_font_family)
                    .justify_between()
                    .items_center()
                    .w(rems(9.0))
                    .child(markdown(reg_id).text_ellipsis().w(rems(2.)))
                    .child(
                        Input::new(&input_state)
                            .appearance(false)
                            .w(rems(9.))
                            .text_color(color)
                            .flex_grow_0(),
                    )
            }
        });

        v_flex()
            .id("cpu-scroll-container")
            .p_2()
            .px_4()
            .h_full()
            .child(
                h_flex()
                    .gap_4()
                    .child(markdown("REGS").text_color(theme.colors.muted_foreground))
                    .child(tabs),
            )
            .border_2()
            .border_color(theme.border)
            .corner_radii(Corners::all(8.0.into()))
            .child(
                v_flex()
                    .min_h_0()
                    .flex_grow_1()
                    .flex_wrap()
                    .w_full()
                    .gap_neg_1()
                    .children(regs),
            )
    }
}
