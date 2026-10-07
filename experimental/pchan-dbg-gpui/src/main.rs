#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::type_complexity)]
#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

#[path = "game-surface.rs"]
pub mod game_surface;

use alloc::borrow::Cow;
use alloc::rc::Rc;
use alloc::sync::Arc;
use core::cell::RefCell;
use core::num::ParseIntError;
use core::ops::Range;
use core::panic::AssertUnwindSafe;
use core::time::Duration;
use core::{cmp, fmt, mem};
use futures_lite::AsyncReadExt;
use futures_lite::io::BufReader;
use pchan_bind::ringbuf::StaticRb;
use pchan_bind::ringbuf::traits::{Consumer, RingBuffer};
use pchan_emu::debug::{Breakpoint, BreakpointKind};
use std::path::PathBuf;
use std::time::Instant;

use bumpalo::Bump;
use gpui::prelude::*;
use gpui::{AppContext, Render, *};
use gpui_base::{Disableable, IndexPath, TextSelectionLayer};
use gpui_component::button::{Button, ButtonVariants, Toggle, ToggleVariants};
use gpui_component::checkbox::Checkbox;
use gpui_component::collapsible::Collapsible;
use gpui_component::input::{Input, InputState};
use gpui_component::menu::DropdownMenu;
use gpui_component::scroll::ScrollableElement;
use gpui_component::select::{
    SearchableVec, Select as SelectView, SelectEvent, SelectItem, SelectState,
};
use gpui_component::separator::Separator;
use gpui_component::spinner::Spinner;
use gpui_component::tab::TabBar;
use gpui_component::table::{Table, TableBody, TableCell, TableHead, TableHeader, TableRow};
use gpui_component::{
    ActiveTheme, Icon, IconName, Root, Sizable, StyledExt, Theme, ThemeRegistry, h_flex, v_flex,
};
use gpui_kit_assets::Assets;
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::cpu::REG_STR;
use pchan_emu::cpu::ops::OpCode;
use pchan_emu::dynarec_v2::emitters::DecodedOp;
use pchan_utils::{hex, hex_pref, init_tracing};

actions!(app, [Quit, SoftReset, HardReset, Step, StepFrame]);

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    init_tracing(pchan_utils::InitTracingArgs {
        stdout:     false,
        file:       true,
        panic_hook: false,
    });
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    #[cfg(feature = "dhat-heap")]
    let _profiler_ptr = (&raw const _profiler).cast_mut();
    let arena: &'static &'static Bump =
        Box::leak(Box::new(Box::leak(Box::new(Bump::new())) as &'static _));

    gpui_platform::application()
        .with_assets(PchanAssets::new())
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(move |cx| {
            gpui_component::init(cx);
            GlobalSelection::init(cx);

            let theme_reg = ThemeRegistry::global_mut(cx);
            let gruvbox = include_str!("./assets/themes/gruvbox.json");
            theme_reg
                .load_themes_from_str(gruvbox)
                .expect("failed to load theme from string");
            let gruvbox = theme_reg.themes().get("Gruvbox Dark").unwrap().clone();

            let theme = Theme::global_mut(cx);
            theme.apply_config(&gruvbox);
            cx.set_window_appearance(Some(WindowAppearance::VibrantDark));

            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);

            cx.on_action(move |_: &Quit, cx| {
                #[cfg(feature = "dhat-heap")]
                unsafe {
                    core::ptr::drop_in_place(_profiler_ptr);
                }
                cx.quit();
            });

            _ = cx.open_window(
                gpui::WindowOptions {
                    window_background: WindowBackgroundAppearance::Blurred,
                    window_decorations: Some(WindowDecorations::Client),
                    titlebar: Some(TitlebarOptions {
                        title:                  None,
                        appears_transparent:    true,
                        traffic_light_position: Some(Point {
                            x: 8.0.into(),
                            y: 8.0.into(),
                        }),
                    }),
                    ..Default::default()
                },
                |win, cx| {
                    let view = Debugger::new(win, cx, arena).unwrap();
                    let view = cx.new(|_| view);
                    Debugger::start_emu_loop(&view, cx);
                    let theme = cx.theme().clone();

                    cx.new(|cx| Root::new(view, win, cx).h_full().bg(theme.background))
                },
            );

            cx.activate(true);
        });
}

#[allow(clippy::struct_field_names)]
struct PchanAssets {
    eject_icon:      &'static [u8],
    disc_3_icon:     &'static [u8],
    rotate_ccw_icon: &'static [u8],
}

impl PchanAssets {
    fn new() -> Self {
        Self {
            eject_icon:      include_bytes!("./assets/eject.svg"),
            disc_3_icon:     include_bytes!("./assets/disc-3.svg"),
            rotate_ccw_icon: include_bytes!("./assets/rotate-ccw.svg"),
        }
    }
}

impl AssetSource for PchanAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        match path {
            "eject.svg" => Ok(Some(Cow::Borrowed(self.eject_icon))),
            "disc-3.svg" => Ok(Some(Cow::Borrowed(self.disc_3_icon))),
            "rotate-ccw.svg" => Ok(Some(Cow::Borrowed(self.rotate_ccw_icon))),
            _ => Assets.load(path),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Assets.list(path)
    }
}

struct Debugger {
    emucx:        Entity<EmuContext>,
    game_surface: Entity<SurfaceState>,

    surface_mode:     Entity<SelectState<Vec<SurfaceMode>>>,
    emu_speed_select: Entity<SelectState<Vec<EmuSpeed>>>,

    cached_reg_names:    [SharedString; 32],
    cpu_control_reg_tab: usize,
    cpu_control_subs:    [Option<Subscription>; 32],

    exec_control_panel_open: bool,
    mips_dump_scroll_handle: VirtualListScrollHandle,
    content_path:            ContentPath,

    memview: Entity<MemviewTable>,
    pc:      u32,
}

enum ContentPath {
    None,
    Disc(PathBuf),
    Exe(PathBuf),
}

pub struct EmuContext {
    emu:                Emu<&'static Bump>,
    runner:             Runner<&'static Bump>,
    running:            bool,
    run_for_one_frame:  bool,
    running_notify:     event_listener::Event,
    renderer:           Arc<pchan_gpu::Renderer>,
    frame_time:         Duration,
    frame_time_limited: Duration,
    real_time_running:  Duration,
    cycles_per_run:     u64,
    start:              Instant,
    speed_limit:        EmuSpeed,
    pc_history:         StaticRb<u32, 50>,
    alloc:              &'static Bump,
}

struct RenderedFrameEvent;
struct ResetEvent;

impl EventEmitter<RenderedFrameEvent> for EmuContext {}
impl EventEmitter<ResetEvent> for EmuContext {}

impl EmuContext {
    pub fn hard_reset(&mut self) -> miette::Result<()> {
        let bios_path = self.emu.boot.bios_path.clone();
        self.renderer.reset();
        self.emu = Emu::new_in(self.alloc);
        self.emu.set_bios_path(bios_path);
        self.emu.load_bios(self.alloc).into_diagnostic()?;
        self.emu.gpu.vram = pchan_emu::gpu::create_vram();
        self.emu.cpu.jump_to_bios();
        self.renderer.connect_emu(&mut self.emu);
        self.emu.tty.set_tracing();

        let mut audio_task = AudioTask::new()?;
        pchan_bind::bind_audio(&mut audio_task, &mut self.emu);
        let audio_stream = audio_task.start()?;
        mem::forget(audio_stream);

        Ok(())
    }

    pub fn execute(&mut self) -> miette::Result<()> {
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.runner.execute(&mut self.emu);
        }))
        .map_err(|_| miette!("emulator panicked"))?;
        self.pc_history.push_overwrite(self.emu.cpu.pc);
        if self
            .emu
            .dbg
            .break_on(self.emu.cpu.pc, BreakpointKind::EXECUTE)
        {
            self.running = false;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EmuSpeed {
    Unlimited,
    Percent(u16),
}

impl SelectItem for EmuSpeed {
    type Value = Self;

    fn title(&self) -> SharedString {
        match self {
            EmuSpeed::Unlimited => "Unlimied".into(),
            EmuSpeed::Percent(p) => format!("{p}%").into(),
        }
    }

    fn value(&self) -> &Self::Value {
        self
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
enum SurfaceMode {
    Background,
    #[default]
    Pane,
}

impl SelectItem for SurfaceMode {
    type Value = Self;

    fn title(&self) -> SharedString {
        match self {
            SurfaceMode::Background => "Background".into(),
            SurfaceMode::Pane => "Pane".into(),
        }
    }

    fn value(&self) -> &Self::Value {
        self
    }
}

impl EmuContext {
    pub fn runner_mode(&self) -> RunnerMode {
        self.runner.mode()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RunnerModeWrapper(Option<RunnerMode>);

impl SelectItem for RunnerModeWrapper {
    type Value = Option<RunnerMode>;

    fn title(&self) -> SharedString {
        match self.0 {
            Some(RunnerMode::Dynarec) => "Dynarec".into(),
            Some(RunnerMode::Interpreter) => "Interpreter".into(),
            None => "Hybrid".into(),
        }
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }
}

use miette::{IntoDiagnostic, miette};
use pchan_emu::run::{Runner, RunnerMode};

use crate::game_surface::{GameSurface, SurfaceState, create_target};

impl Debugger {
    pub fn new(window: &mut Window, cx: &mut App, alloc: &'static Bump) -> miette::Result<Self> {
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
        let gpu = cx.foreground_executor().block_on(gpu).into_diagnostic()?;

        let mut dp = gpu.display_uniforms.app.lock().unwrap();
        dp.screen_rect.x = 320;
        dp.screen_rect.y = 240;
        drop(dp);

        gpu.connect_emu(&mut emu);
        let gpu = Arc::new(gpu);
        gpu.clone().start();

        let (target, target_buf) = create_target(&gpu, &gpu.display_uniforms.app.lock().unwrap());

        let cached_reg_names = core::array::from_fn(|reg| {
            let reg = match reg as u8 {
                0 => "0",
                pchan_emu::cpu::FP => "fp",
                other => REG_STR[other as usize],
            };
            format!("${reg}").into()
        });

        let emucx = cx.new(|_| EmuContext {
            emu,
            renderer: gpu.clone(),
            running: false,
            running_notify: event_listener::Event::new(),
            runner: Runner::new_in(alloc).with_config(pchan_emu::run::RunnerConfig {
                force_mode: Some(pchan_emu::run::RunnerMode::Dynarec),
            }),
            frame_time: Duration::ZERO,
            frame_time_limited: Duration::ZERO,
            start: Instant::now(),
            cycles_per_run: 0,
            speed_limit: EmuSpeed::Percent(100),
            real_time_running: Duration::ZERO,
            run_for_one_frame: false,
            alloc,
            pc_history: StaticRb::default(),
        });

        let surface_state = cx.new(|_| SurfaceState::new(target.clone(), target_buf.clone()));

        let memview = cx.new(|cx| {
            let mut memview_scroll = VirtualListScrollHandle::new();
            memview_scroll.scroll_to(cx, 0x0000_0000, ScrollStrategy::Top);
            MemviewTable {
                scroll:   memview_scroll,
                emucx:    emucx.clone(),
                editing:  None,
                selected: None,
            }
        });

        let emu_speed_select = cx.new(|cx| {
            SelectState::new(
                vec![
                    EmuSpeed::Percent(100),
                    EmuSpeed::Percent(150),
                    EmuSpeed::Percent(200),
                    EmuSpeed::Unlimited,
                ],
                Some(IndexPath::default()),
                window,
                cx,
            )
        });
        let surface_mode = cx.new(|cx| {
            SelectState::new(
                vec![SurfaceMode::Pane, SurfaceMode::Background],
                Some(IndexPath::default()),
                window,
                cx,
            )
        });

        cx.subscribe(&emu_speed_select, {
            let emucx = emucx.clone();
            move |_, event: &SelectEvent<Vec<EmuSpeed>>, cx| match event {
                SelectEvent::Confirm(value) => {
                    let Some(selected_value) = value else {
                        return;
                    };
                    emucx.update(cx, |emucx, _| emucx.speed_limit = *selected_value)
                }
            }
        })
        .detach();

        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);

        cx.on_action({
            let emucx = emucx.clone();
            move |_: &Quit, cx| {
                emucx.read(cx).emu.dma.dump_cdrom_data();
                cx.quit();
            }
        });

        cx.on_action::<HardReset>({
            let emucx = emucx.clone();
            move |_, cx| {
                emucx.update(cx, |emucx, _| emucx.hard_reset()).unwrap();
            }
        });

        cx.on_action::<Step>({
            let emucx = emucx.clone();
            let surface_state = surface_state.clone();
            move |_, cx| {
                surface_state.update(cx, |surface, _| {
                    surface.clear(&gpu);
                });
                emucx.update(cx, |emucx, _| {
                    let _ = emucx.execute();
                    emucx.running_notify.notify(usize::MAX);
                });
            }
        });

        cx.on_action::<StepFrame>({
            let emucx = emucx.clone();
            move |_, cx| {
                emucx.update(cx, |emucx, _| {
                    emucx.run_for_one_frame = true;
                    emucx.running = true;
                    emucx.running_notify.notify(usize::MAX);
                });
            }
        });

        Ok(Self {
            emucx,
            game_surface: surface_state,
            surface_mode,

            memview,

            cached_reg_names,
            cpu_control_subs: [const { None }; 32],
            cpu_control_reg_tab: 0,
            exec_control_panel_open: true,
            content_path: ContentPath::None,
            mips_dump_scroll_handle: VirtualListScrollHandle::new(),
            pc: 0,
            emu_speed_select,
        })
    }

    pub fn start_emu_loop(this: &Entity<Debugger>, cx: &mut App) {
        let surface_state = this.read(cx).game_surface.clone();
        let emucx = this.read(cx).emucx.clone();
        let dbg = this.clone();
        cx.spawn_with_priority(Priority::High, {
            async move |cx| {
                loop {
                    match Self::emu_loop(cx, &emucx, &surface_state, &dbg).await {
                        Ok(_) => {}
                        Err(_) => {
                            emucx.update(cx, |emucx, _| {
                                emucx.running = false;
                                _ = emucx.hard_reset();
                            });
                        }
                    }
                }
            }
        })
        .detach();
    }

    async fn emu_loop(
        cx: &mut AsyncApp,
        emucx: &Entity<EmuContext>,
        surface_state: &Entity<SurfaceState>,
        debugger: &Entity<Debugger>,
    ) -> miette::Result<()> {
        let mut yield_time = Duration::ZERO;
        let mut first_frame_rendered = false;

        loop {
            let run_listener = cx.read_entity(emucx, |emucx, _| match emucx.running {
                false => Some(emucx.running_notify.listen()),
                true => None,
            });
            if let Some(run_listener) = run_listener {
                debugger.update(cx, |_, cx| {
                    cx.notify();
                });
                run_listener.await;
                emucx.update(cx, |emucx, _| {
                    emucx.start = Instant::now();
                    emucx.cycles_per_run = 0;
                    emucx.real_time_running = Duration::ZERO;
                });
            }
            let start = Instant::now();
            let old_cycles = cx.read_entity(emucx, |emucx, _| emucx.emu.cpu.cycles);
            emucx.update(cx, |emucx, cx| -> miette::Result<()> {
                if emucx.running {
                    surface_state.as_mut(cx).start_display_draw(&emucx.renderer);

                    while !emucx.emu.consume_vblank_signal() {
                        emucx.execute()?;
                        if !emucx.running {
                            break;
                        }
                    }

                    let surface_state = surface_state.as_mut(cx);
                    surface_state.wait_for_display_draw(&emucx.renderer);
                    surface_state.start_convert_render(&emucx.renderer);
                    drop(surface_state);
                    cx.emit(RenderedFrameEvent);

                    if emucx.run_for_one_frame {
                        emucx.run_for_one_frame = false;
                        emucx.running = false;
                    }
                }
                Ok(())
            })?;
            let emu_frame_time = start.elapsed();
            yield_time += emu_frame_time;

            let frame_limit = match cx.read_entity(emucx, |emucx, _| emucx.speed_limit) {
                EmuSpeed::Unlimited => Duration::ZERO,
                EmuSpeed::Percent(p) => Duration::from_micros(16_667 * 100 / u64::from(p)),
            };

            let yield_max = cmp::max(frame_limit * 4, Duration::from_micros(16_667 * 4));
            if yield_time > yield_max {
                debugger.update(cx, |_, cx| {
                    cx.notify();
                });
                yield_time = Duration::ZERO;
                futures_lite::future::yield_now().await;
            }

            let (frame_time, frame_limit) = emucx.update(cx, |emucx, _| {
                let frame_time = start.elapsed();
                emucx.frame_time = frame_time;
                yield_time += frame_time;

                // cycles get reset when doing a hard reset, so they are not
                // strictly monotonic
                let delta_cycles = emucx.emu.cpu.cycles.saturating_sub(old_cycles);
                emucx.cycles_per_run += delta_cycles;

                emucx.frame_time_limited = frame_time.max(frame_limit);
                if first_frame_rendered {
                    emucx.real_time_running += emu_frame_time.max(frame_limit);
                } else {
                    emucx.cycles_per_run = 0;
                    emucx.real_time_running = Duration::ZERO;
                    first_frame_rendered = true;
                }

                (frame_time, frame_limit)
            });
            println!(
                "emu: {}ms, emu+gpui: {}ms",
                emu_frame_time.as_millis(),
                frame_time.as_millis()
            );

            let yielded = spin_sleep(cx, frame_limit.saturating_sub(frame_time)).await;
            if yielded {
                yield_time = Duration::ZERO;
            }
        }
    }
}

async fn spin_sleep(cx: &AsyncApp, duration: Duration) -> bool {
    let sleep_for = duration.saturating_sub(Duration::from_millis(3));
    let deadline = Instant::now() + duration;

    let mut yielded = false;
    if sleep_for > Duration::ZERO {
        yielded = true;
        cx.background_executor()
            .spawn_with_priority(Priority::Low, cx.background_executor().timer(sleep_for))
            .await;
    }
    loop {
        if Instant::now() > deadline {
            return yielded;
        }
        core::hint::spin_loop();
    }
}

impl Render for Debugger {
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl gpui::prelude::IntoElement {
        let window_height = window.viewport_size().height;

        let _theme = cx.theme();

        let surface = self.game_surface(cx);
        let emucx = self.emucx.read(cx);
        let surface_mode = self
            .surface_mode
            .read(cx)
            .selected_value()
            .copied()
            .unwrap_or(SurfaceMode::Background);

        v_flex()
            .child(TextSelectionLayer)
            .h(window_height)
            .child(self.header(
                window,
                cx,
                emucx.frame_time,
                emucx.frame_time_limited,
                emucx.cycles_per_run,
                emucx.real_time_running,
            ))
            .child(
                div()
                    .h_full()
                    .bg(transparent_white())
                    .when(surface_mode == SurfaceMode::Background, |this| {
                        this.child(div().absolute().w_full().h_full().child(surface))
                    })
                    .child(self.debugger_ui(window, cx).w_full().h_full()),
            )
    }
}

impl Debugger {
    fn game_surface(&self, cx: &mut impl AppContext) -> GameSurface {
        GameSurface::new(
            "game-surface",
            cx.read_entity(&self.emucx, |emucx, _| emucx.renderer.clone()),
            self.game_surface.clone(),
        )
        .w_full()
        .h_full()
        .min_h_4()
        .min_w_4()
    }

    fn header(
        &self,
        win: &mut Window,
        cx: &mut Context<Debugger>,
        frame_time: Duration,
        frame_time_limited: Duration,
        cycles: u64,
        real_time: Duration,
    ) -> impl IntoElement {
        let sim_time_ms = cycles * 1000 / u64::from(pchan_emu::cpu::Cpu::CLOCK);
        let sim_time_s = sim_time_ms as f64 / 1000.0;
        let real_time_s = real_time.as_millis() as f64 / 1000.0;
        let drift = ((sim_time_s - real_time_s) / sim_time_s) * 100.0;

        let speed_percent = 16.667 / (frame_time_limited.as_micros() as f64 / 1000.0) * 100.0;

        let runner_mode_select = win.use_state(cx, |win, cx| {
            SelectState::new(
                SearchableVec::new(vec![
                    RunnerModeWrapper(Some(RunnerMode::Dynarec)),
                    RunnerModeWrapper(Some(RunnerMode::Interpreter)),
                    RunnerModeWrapper(None),
                ]),
                Some(IndexPath::default()),
                win,
                cx,
            )
            .searchable(true)
        });
        let view = cx.entity();
        win.use_state(cx, |win, cx| {
            cx.subscribe_in(
                &runner_mode_select,
                win,
                move |_, _, ev: &SelectEvent<SearchableVec<RunnerModeWrapper>>, _, cx| {
                    if let SelectEvent::Confirm(Some(mode)) = ev {
                        view.read(cx).emucx.clone().update(cx, |emucx, _| {
                            #[allow(clippy::needless_update)]
                            emucx.runner.set_config(pchan_emu::run::RunnerConfig {
                                force_mode: *mode,
                                ..emucx.runner.config()
                            })
                        });
                    }
                },
            )
            .detach();
        });

        let theme = cx.theme();

        h_flex()
            .text_sm()
            .bg(theme.title_bar)
            .px_4()
            .pl(rems(4.5))
            .py_1()
            .gap_2()
            .items_center()
            .border_color(theme.title_bar_border)
            .border_b_1()
            .text_color(theme.table_head_foreground)
            .font_family(&theme.mono_font_family)
            .child("🐷🎗️ P-ちゃん")
            .child(
                div().child(
                    SelectView::new(&self.emu_speed_select)
                        .title_prefix("Speed: ")
                        .items_center()
                        .small()
                        .h_6()
                        .min_h_0()
                        .min_w_0()
                        .flex_shrink_1()
                        .flex_grow_0(),
                ),
            )
            .child(
                div().child(
                    SelectView::new(&self.surface_mode)
                        .title_prefix("Game: ")
                        .items_center()
                        .small()
                        .h_6()
                        .min_h_0()
                        .min_w_0()
                        .flex_shrink_1()
                        .flex_grow_0(),
                ),
            )
            .child(
                div().child(
                    SelectView::new(&runner_mode_select)
                        .title_prefix("CPU: ")
                        .items_center()
                        .small()
                        .h_6()
                        .min_h_0()
                        .min_w_0()
                        .flex_shrink_1()
                        .flex_grow_0(),
                ),
            )
            .child(div().flex_grow_1())
            .child(format!("frame: {:02}ms", frame_time.as_millis()))
            .child(Separator::vertical())
            .child(match cycles {
                ..1_000 => format!("cycles: {cycles}"),
                1_000..1_000_000 => format!("cycles: {}k", cycles / 1_000),
                1_000_000..1_000_000_000 => format!("cycles: {}mil", cycles / 1_000),
                1_000_000_000.. => format!("cycles: {}bil", cycles / 1_000),
            })
            .child(Separator::vertical())
            .child(
                h_flex()
                    .child(div().text_color(theme.colors.yellow).child("sim"))
                    .child("/")
                    .child(div().text_color(theme.colors.blue).child("real"))
                    .child(" time: ")
                    .child(
                        div()
                            .text_color(theme.colors.yellow)
                            .child(format!("{sim_time_s:01.2}s")),
                    )
                    .child("/")
                    .child(
                        div()
                            .text_color(theme.colors.blue)
                            .child(format!("{real_time_s:01.2}s")),
                    )
                    .child(" drift: ")
                    .child(div().child(format!("{drift:.2}%")).text_color(match drift {
                        ..0.0 => theme.colors.yellow,
                        _ => theme.colors.blue,
                    })),
            )
            .child(format!("{speed_percent:.2}%"))
    }

    fn memview_jumpbar(&mut self, win: &mut Window, cx: &mut Context<Self>) -> Input {
        let memview = self.memview.clone();
        let input = win.use_state(cx, |win, cx| {
            cx.subscribe_in(
                &memview,
                win,
                move |input: &mut HexInputState, _, edit, win, cx| {
                    input.input.update(cx, |input, cx| {
                        input.set_value(hex(edit.address).as_str(), win, cx);
                    });
                    cx.notify();
                },
            )
            .detach();

            HexInputState::new::<true, _>(None, win, cx, {
                let memview = memview.clone();
                move |value, cx| -> Option<()> {
                    let value = value?;
                    memview.update(cx, |memview, cx| {
                        memview
                            .scroll
                            .scroll_to(cx, u64::from(value) / 16, ScrollStrategy::Top);
                    });
                    None
                }
            })
        });
        let theme = cx.theme();
        Input::new(&input.read(cx).input)
            .font_family(&theme.mono_font_family)
            .prefix("Jump to: ")
    }

    fn debugger_ui(
        &mut self,
        win: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + Styled {
        let theme = cx.theme().clone();
        let surface_mode = self
            .surface_mode
            .read(cx)
            .selected_value()
            .copied()
            .unwrap_or(SurfaceMode::Background);
        let surface = self.game_surface(cx).clone();
        let emucx = self.emucx.clone();
        let show_vram = emucx
            .read(cx)
            .renderer
            .display_uniforms
            .app
            .lock()
            .unwrap()
            .dp_debug;

        v_flex()
            .id("main")
            .w_full()
            .h_full()
            .relative()
            .text_color(theme.foreground)
            .p_4()
            .gap_2()
            .bg(transparent_white())
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .items_start()
                    .flex_grow_1()
                    .text_sm()
                    .child(
                        panel(&theme)
                            .v_flex()
                            .flex_grow_1()
                            .min_w(rems(24.))
                            .when(self.exec_control_panel_open, |this| this.min_h_full())
                            .gap_2()
                            .child(
                                h_flex()
                                    .justify_between()
                                    .gap_2()
                                    .child(
                                        self.execution_header(cx, &theme).min_w_0().flex_grow_1(),
                                    )
                                    .child(
                                        Button::new("toggle1")
                                            .icon(IconName::ChevronDown)
                                            .ghost()
                                            .small()
                                            .on_click({
                                                cx.listener(move |this, _, _, cx| {
                                                    this.exec_control_panel_open.toggle();
                                                    cx.notify();
                                                })
                                            }),
                                    ),
                            )
                            .when(self.exec_control_panel_open, |this| {
                                this.child(
                                    Collapsible::new()
                                        .open(self.exec_control_panel_open)
                                        .flex()
                                        .flex_grow_1()
                                        .h_full()
                                        .w_full()
                                        .content(
                                            self.execution_control(cx, &theme)
                                                .w_full()
                                                .min_h_0()
                                                .flex_grow_1(),
                                        ),
                                )
                            }),
                    )
                    .when(surface_mode == SurfaceMode::Pane, |this| {
                        this.child(
                            panel(&theme)
                                .v_flex()
                                .flex_grow_1()
                                .w_full()
                                .h_full()
                                .min_w_4()
                                .min_h_4()
                                .gap_2()
                                .child(
                                    Checkbox::new("display-vram-checkbox")
                                        .checked(show_vram)
                                        .label("VRAM")
                                        .on_click({
                                            let surface = surface.clone();
                                            cx.listener(move |_, toggled, _, cx| {
                                                surface.set_vram_view(*toggled);
                                                cx.notify();
                                            })
                                        }),
                                )
                                .child(surface),
                        )
                    })
                    .child(
                        self.breakpoints(win, cx, &theme)
                            .max_w(rems(20.))
                            .min_w(rems(18.))
                            .flex_grow_1()
                            .h_full(),
                    ),
            )
            // bottom panel
            .child(
                panel(&theme)
                    .h_flex()
                    .text_sm()
                    .flex_grow_1()
                    .max_h(rems(16.))
                    .min_h_0()
                    .w_full()
                    .gap_2()
                    .text_sm()
                    .child(
                        v_flex()
                            .h_full()
                            .min_h_0()
                            .min_w_64()
                            .gap_2()
                            .child(
                                div()
                                    .h_flex()
                                    .items_center()
                                    .flex_grow_0()
                                    .gap_2()
                                    .child(
                                        div().child("Registers").text_color(theme.muted_foreground),
                                    )
                                    .child(self.cpu_control_tabbar(cx)),
                            )
                            .child(self.cpu_controls(win, cx).min_h_0().min_w_0().flex_grow_1()),
                    )
                    .child(
                        v_flex()
                            .flex_grow_1()
                            .h_full()
                            .min_h_0()
                            .gap_2()
                            .child(
                                h_flex().h_8().child(
                                    h_flex()
                                        .gap_2()
                                        .child("Memory")
                                        .text_color(theme.muted_foreground)
                                        .child(self.memview_jumpbar(win, cx).w_64())
                                        .child(div().w(rems(6.)))
                                        .child(div().child("Ascii")),
                                ),
                            )
                            .child(
                                panel(&theme)
                                    .flex_grow_1()
                                    .min_h_0()
                                    .child(self.memview.clone()),
                            ),
                    )
                    .child(
                        v_flex()
                            .h_full()
                            .w(rems(11.5))
                            .min_h_0()
                            .gap_2()
                            .child(
                                h_flex().h_8().child(
                                    h_flex()
                                        .gap_2()
                                        .child("Mem Inspector")
                                        .text_color(theme.muted_foreground),
                                ),
                            )
                            .child(
                                panel(&theme)
                                    .flex_grow_1()
                                    .w_full()
                                    .min_h_0()
                                    .px_0()
                                    .child(self.mem_inspector(cx, &theme)),
                            ),
                    ),
            )
    }

    fn breakpoints(
        &self,
        win: &mut Window,
        cx: &mut Context<Self>,
        theme: &Theme,
    ) -> impl IntoElement + Styled {
        fn bp_toggle(
            id: impl Into<ElementId>,
            address: u32,
            kind: BreakpointKind,
            bp: Breakpoint,
            label: &'static str,
            cx: &Context<Debugger>,
        ) -> Toggle {
            Toggle::new(id)
                .checked(bp.kind.contains(kind))
                .label(label)
                .aspect_square()
                .min_w_0()
                .w_8()
                .outline()
                .flex_grow_0()
                .on_click(cx.listener(move |view, ev, _, cx| {
                    view.emucx.update(cx, |emucx, _| {
                        let Some(bp) = emucx.emu.dbg.breakpoints.get_mut(&address) else {
                            return;
                        };
                        match *ev {
                            true => {
                                bp.kind |= kind;
                            }
                            false => {
                                bp.kind = bp.kind.difference(kind);
                            }
                        }
                    })
                }))
        }

        let add_bp_input = win.use_state(cx, |win, cx| {
            InputState::new(win, cx).placeholder("Add breakpoint...")
        });

        let emucx = self.emucx.clone();
        win.use_state(cx, {
            let add_bp_input = add_bp_input.clone();
            move |win, cx| {
                use gpui_component::input::InputEvent;

                cx.subscribe_in(
                    &add_bp_input,
                    win,
                    move |(), state, ev: &InputEvent, win, cx| {
                        let _ = state.update(cx, |state, cx| -> miette::Result<()> {
                            let InputEvent::PressEnter { .. } = ev else {
                                return Ok(());
                            };
                            let address = parse_hex_word(&state.value()).into_diagnostic()?;
                            let address = address & 0x1fff_ffff;
                            state.clean(win, cx);

                            emucx.update(cx, |emucx, _| {
                                emucx.emu.dbg.breakpoints.insert(
                                    address,
                                    Breakpoint {
                                        address,
                                        kind: BreakpointKind::EXECUTE,
                                        enabled: true,
                                    },
                                )
                            });
                            Ok(())
                        });
                    },
                )
                .detach();
            }
        });
        let breakpoints = self
            .emucx
            .read(cx)
            .emu
            .dbg
            .breakpoints
            .iter()
            .map(|(address, bp)| {
                let address = *address;
                let idn = u64::from(address);
                panel(theme)
                    .h_flex()
                    .gap_2()
                    .child(
                        div()
                            .child(SharedString::from(hex(address).as_str()))
                            .font_family(&theme.mono_font_family),
                    )
                    .child(bp_toggle(
                        ElementId::NamedInteger("bp-read".into(), idn),
                        address,
                        BreakpointKind::READ,
                        *bp,
                        "R",
                        cx,
                    ))
                    .child(bp_toggle(
                        ElementId::NamedInteger("bp-write".into(), idn),
                        address,
                        BreakpointKind::WRITE,
                        *bp,
                        "W",
                        cx,
                    ))
                    .child(bp_toggle(
                        ElementId::NamedInteger("bp-execute".into(), idn),
                        address,
                        BreakpointKind::EXECUTE,
                        *bp,
                        "X",
                        cx,
                    ))
                    .child(div().flex_grow_1())
                    .child(
                        Button::new(ElementId::NamedInteger("bp-delete".into(), idn))
                            .outline()
                            .icon(IconName::Close)
                            .on_click(cx.listener(move |state, _, _, cx| {
                                state.emucx.update(cx, |emucx, _| {
                                    emucx.emu.dbg.remove_breakpoint(address);
                                })
                            })),
                    )
            });
        panel(theme)
            .v_flex()
            .gap_2()
            .child(
                div()
                    .child("Breakpoints")
                    .text_color(theme.muted_foreground),
            )
            .child(Input::new(&add_bp_input))
            .child(
                div()
                    .v_flex()
                    .gap_2()
                    .w_full()
                    .h_1_2()
                    .overflow_y_scrollbar()
                    .min_h_0()
                    .children(breakpoints),
            )
            .child(
                v_flex().w_full().overflow_y_scrollbar().min_h_0().children(
                    self.emucx
                        .read(cx)
                        .pc_history
                        .iter()
                        .rev()
                        .copied()
                        .enumerate()
                        .map(|(i, pc)| {
                            sel_text_keyed(
                                ElementId::NamedInteger("pc-history-addr".into(), i as u64),
                                hex(pc).as_str(),
                            )
                            .font_family(&theme.mono_font_family)
                        }),
                ),
            )
    }
}

fn panel(theme: &Theme) -> Div {
    div()
        .border_2()
        .bg(theme.background)
        .border_color(theme.border)
        .corner_radii(Corners::all(8.0.into()))
        .overflow_hidden()
        .p_2()
}

fn parse_hex_word(str: &str) -> Result<u32, ParseIntError> {
    if str == "0x" {
        return Ok(0);
    }
    let str = str.trim_prefix("0x");
    if let Some((a, b)) = str.split_once('_') {
        let a = parse_hex_word(a)?;
        let b = parse_hex_word(b)?;
        Ok(a << 16 | b)
    } else {
        u32::from_str_radix(str, 16)
    }
}

impl Debugger {
    fn cpu_control_tabbar(&mut self, cx: &mut Context<Debugger>) -> impl IntoElement + Styled {
        TabBar::new("segmented-tabs")
            .min_h_0()
            .segmented()
            .selected_index(self.cpu_control_reg_tab)
            .cursor_pointer()
            .on_click(cx.listener(|view, index, _, cx| {
                view.cpu_control_reg_tab = *index;
                cx.notify();
            }))
            .children(vec!["CPU", "COP0", "GTE"])
    }

    fn cpu_controls(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Debugger>,
    ) -> impl IntoElement + Styled {
        let theme = cx.theme().clone();

        let gpr = self.emucx.read(cx).emu.cpu.gpr.clone();
        let gpr = gpr.iter().copied().enumerate().map({
            |(r, value)| {
                use gpui_component::input::InputEvent;

                let reg_id = &self.cached_reg_names[r];
                let input_state = window.use_keyed_state(reg_id.clone(), cx, |win, cx| {
                    let reg_value: SharedString = hex(value).to_string().into();
                    InputState::new(win, cx)
                        .default_value(reg_value)
                        .validate(|value, _| parse_hex_word(value).is_ok())
                });

                match self.cpu_control_subs[r] {
                    Some(_) => {}
                    None => {
                        let sub = cx.subscribe_in(
                            &input_state,
                            window,
                            move |view, input_state, event, win, cx| {
                                if let InputEvent::PressEnter { .. } | InputEvent::Blur = event {
                                    let reg_value =
                                        match parse_hex_word(&input_state.read(cx).value()) {
                                            Ok(reg_value) => {
                                                view.emucx.update(cx, |emucx, _| {
                                                    emucx.emu.cpu.gpr[r] = reg_value
                                                });
                                                reg_value
                                            }
                                            // TODO: handle error
                                            Err(_err) => return,
                                        };
                                    input_state.update(cx, |state, cx| {
                                        state.set_value(hex(reg_value).as_str(), win, cx);
                                    })
                                }
                            },
                        );
                        self.cpu_control_subs[r] = Some(sub);
                    }
                }

                if !input_state.focus_handle(cx).is_focused(window) {
                    input_state.update(cx, |state, cx| {
                        state.set_value(hex(value).as_str(), window, cx);
                    });
                }

                let color = match value {
                    0 => theme.colors.muted_foreground,
                    _ => theme.colors.foreground,
                };

                div()
                    .flex()
                    .id(reg_id.clone())
                    .font_family(&theme.mono_font_family)
                    .justify_between()
                    .items_center()
                    .min_w_0()
                    .h(rems(1.))
                    .w(rems(9.0))
                    .child(div().child(reg_id.clone()).text_ellipsis().w(rems(2.)))
                    .child(
                        Input::new(&input_state)
                            .appearance(input_state.focus_handle(cx).is_focused(window))
                            .w(rems(9.))
                            .text_color(color)
                            .flex_grow_0(),
                    )
            }
        });

        panel(&theme)
            .v_flex()
            .id("cpu-scroll-container")
            .gap_2()
            .min_w_0()
            .child(
                div()
                    .gap_1()
                    .v_flex()
                    .flex_wrap()
                    .min_w_0()
                    .overflow_x_scrollbar()
                    .flex_grow_1()
                    .children(gpr),
            )
    }

    fn instructions_list(
        &mut self,
        cx: &mut Context<Debugger>,
        theme: &Theme,
    ) -> impl IntoElement + Styled {
        let entity = cx.entity().clone();
        let theme = theme.clone();
        let pc = self.emucx.read(cx).emu.cpu.pc;

        if pc != self.pc {
            self.mips_dump_scroll_handle.scroll_to(
                cx,
                u64::from(pc) / 4,
                ScrollStrategy::TopOffset(4),
            );
            self.pc = pc;
        }

        v_flex()
            .child(sel_text(format!("$pc: {}", hex(pc))).font_family(&theme.mono_font_family))
            .child(
                VirtualList::new(
                    "mips-dump-list",
                    u64::from(u32::MAX) / 4,
                    move |idx, _, cx| {
                        let address_label: SharedString = "mips-dump-address".into();

                        let view = entity.read(cx);
                        let addr = idx as u32 * 4;
                        let instr = view.emucx.read(cx).emu.fastmem_read::<OpCode>(addr);
                        let instr = instr
                            .map(DecodedOp::new)
                            .map_or(Cow::Borrowed("N/A"), |instr| Cow::Owned(format!("{instr}")));
                        let is_pc = pc & 0x1fff_ffff == addr & 0x1fff_ffff;
                        h_flex()
                            .w_full()
                            .whitespace_nowrap()
                            .overflow_hidden()
                            .font_family(&theme.mono_font_family)
                            .bg(theme.foreground.opacity(if idx.is_multiple_of(2) {
                                0.0
                            } else {
                                0.08
                            }))
                            .child(
                                sel_text_keyed(
                                    ElementId::NamedInteger(
                                        "mipds-dump-list-address-column".into(),
                                        u64::from(addr),
                                    ),
                                    hex(addr).to_string(),
                                )
                                .opacity(0.5),
                            )
                            .child(
                                div()
                                    .text_center()
                                    .min_w_4()
                                    .w_4()
                                    .when(is_pc, |this| this.child(">")),
                            )
                            .child(sel_text_keyed(
                                ElementId::NamedInteger(address_label, idx),
                                instr,
                            ))
                            .when(is_pc, |this| this.text_color(theme.colors.info))
                            .h_4()
                    },
                )
                .track_scroll(&self.mips_dump_scroll_handle),
            )
    }

    fn open_content_button(cx: &mut Context<Debugger>, _theme: &Theme) -> Button {
        Button::new("content-path-button")
            .secondary()
            .on_click(cx.listener(|_, _, _, cx| {
                let path_recv = cx.prompt_for_paths(PathPromptOptions {
                    files:       true,
                    directories: false,
                    multiple:    false,
                    prompt:      Some("Open disc file (.bin, .cue)".into()),
                });
                cx.spawn(async move |view, cx| -> miette::Result<()> {
                    let Some(view) = view.upgrade() else {
                        return Ok(());
                    };
                    let res = path_recv.await.into_diagnostic()?;
                    let res = res.map_err(|err| miette!("error: {err}"))?;
                    let mut res = res.ok_or_else(|| miette!("no file selected"))?;
                    let content_path = res
                        .pop()
                        .ok_or_else(|| miette!("expected at least one disc path"))?;

                    match content_path
                        .extension()
                        .map(|ext| ext.to_string_lossy())
                        .as_deref()
                    {
                        Some("bin" | "cue") => {
                            view.update(cx, move |view, cx| -> miette::Result<()> {
                                view.emucx.update(cx, |emucx, _| -> miette::Result<()> {
                                    let fsm = emucx
                                        .emu
                                        .open_disc(&content_path, true)
                                        .into_diagnostic()?;
                                    emucx
                                        .emu
                                        .advance_open_disc(&content_path, fsm, true)
                                        .into_diagnostic()?;
                                    Ok(())
                                })?;
                                view.content_path = ContentPath::Disc(content_path);
                                Ok(())
                            })?;
                        }
                        _ => {
                            let mut file = BufReader::new(
                                async_fs::File::open(&content_path)
                                    .await
                                    .into_diagnostic()?,
                            );
                            let mut exe = Vec::new();
                            file.read_to_end(&mut exe).await.into_diagnostic()?;

                            view.update(cx, move |view, cx| -> miette::Result<()> {
                                view.emucx.update(cx, |emucx, _| -> miette::Result<()> {
                                    emucx.hard_reset()?;
                                    emucx.emu.sideload_exe(&exe).into_diagnostic()?;
                                    Ok(())
                                })?;
                                view.content_path = ContentPath::Exe(content_path);
                                Ok(())
                            })?;
                        }
                    }

                    Ok(())
                })
                .detach();
            }))
    }

    fn disc_buttons(
        &mut self,
        cx: &mut Context<Debugger>,
        theme: &Theme,
    ) -> impl IntoElement + Styled {
        let open_content = Debugger::open_content_button(cx, theme);
        match &self.content_path {
            ContentPath::None => h_flex().flex_grow_1().child(
                open_content
                    .label("Load Content")
                    .w_full()
                    .icon(Icon::empty().path("disc-3.svg")),
            ),
            ContentPath::Disc(path) | ContentPath::Exe(path) => h_flex()
                .min_w_0()
                .gap_2()
                .w_full()
                .child(Button::new("eject-disc-button").icon(Icon::empty().path("eject.svg")))
                .child(
                    div().min_w_0().flex_grow_1().child(
                        open_content.w_full().text_ellipsis().flex().label(
                            path.file_name()
                                .map_or(Cow::Borrowed("Unknown"), |f| f.to_string_lossy()),
                        ),
                    ),
                ),
        }
    }

    fn execution_header(
        &mut self,
        cx: &mut Context<Debugger>,
        theme: &Theme,
    ) -> impl IntoElement + Styled {
        let emucx = self.emucx.read(cx);
        h_flex()
            .gap_2()
            .items_center()
            .child(
                Button::new("reset-button")
                    .icon(Icon::empty().path("rotate-ccw.svg"))
                    .dropdown_menu(|menu, _, _| {
                        menu.menu("Hard Reset", Box::new(HardReset))
                            .menu("Soft Reset", Box::new(SoftReset))
                    }),
            )
            .child(
                Button::new("run-button")
                    .w_24()
                    .cursor_pointer()
                    .label(match emucx.running {
                        true => "Pause",
                        false => "Run",
                    })
                    .children(
                        emucx.running.then_some(
                            Spinner::new()
                                .icon(IconName::LoaderCircle)
                                .color(theme.muted_foreground),
                        ),
                    )
                    .on_click(cx.listener(|view, _, _, cx| {
                        view.emucx.update(cx, |emucx, _| {
                            emucx.running.toggle();
                            emucx.running_notify.notify(usize::MAX);
                        })
                    })),
            )
            .child(self.disc_buttons(cx, theme))
    }

    fn execution_control(
        &mut self,
        cx: &mut Context<Debugger>,
        theme: &Theme,
    ) -> impl IntoElement + Styled {
        let emucx = self.emucx.read(cx);
        div()
            .v_flex()
            .gap_2()
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("step-btn")
                            .label(match emucx.runner_mode() {
                                RunnerMode::Dynarec => "Step block",
                                RunnerMode::Interpreter => "Step instr.",
                            })
                            .disabled(emucx.running)
                            .on_click(|_, win, cx| {
                                win.dispatch_action(Box::new(Step), cx);
                            }),
                    )
                    .child(
                        Button::new("step-frame-btn")
                            .label("Step frame")
                            .disabled(emucx.running)
                            .on_click(|_, win, cx| {
                                win.dispatch_action(Box::new(StepFrame), cx);
                            }),
                    ),
            )
            .child(Separator::horizontal())
            .child(self.instructions_list(cx, theme).flex_grow_1().min_h_0())
    }

    fn mem_inspector(
        &mut self,
        cx: &mut Context<Debugger>,
        _theme: &Theme,
    ) -> impl IntoElement + Styled {
        fn table_row<T: Copy + core::fmt::Debug>(
            type_label: &'static str,
            emucx: &Entity<EmuContext>,
            cx: &mut Context<Debugger>,
            address: Option<u32>,
        ) -> TableRow {
            TableRow::new()
                .gap_2()
                .child(TableCell::new().child(type_label).min_w_0().w(rems(4.)))
                .child(TableCell::new().child(get_value::<T>(emucx, cx, address)))
        }

        fn get_value<T: Copy + core::fmt::Debug>(
            emucx: &Entity<EmuContext>,
            cx: &mut Context<Debugger>,
            address: Option<u32>,
        ) -> SharedString {
            let Some(address) = address else {
                return " ".into();
            };
            match emucx.read(cx).emu.try_read_pure::<T>(address) {
                Ok(value) => format!("{value:?}").into(),
                Err(_) => "N/A".into(),
            }
        }

        #[derive(Clone, Copy)]
        struct Ascii([u8; 4]);

        impl core::fmt::Debug for Ascii {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut word = self.0;
                for byte in &mut word {
                    match *byte {
                        ..=0x1f | 0x7f.. => {
                            *byte = b'.';
                        }
                        _ => {}
                    }
                }

                let word = core::str::from_utf8(&word).expect("impossible");
                write!(f, "{word:?}")
            }
        }

        let selected = self.memview.read(cx).selected;

        div()
            .w_full()
            .h_full()
            .min_w_0()
            .overflow_y_scrollbar()
            .child(
                div().overflow_x_scrollbar().w_full().h_full().child(
                    Table::new()
                        .child(
                            TableHeader::new().child(
                                TableRow::new()
                                    .child(TableHead::new().child("Type").min_w_0().w(rems(4.)))
                                    .child(TableHead::new().child("Value")),
                            ),
                        )
                        .child(
                            TableBody::new()
                                .w_full()
                                .child(table_row::<u32>("u32", &self.emucx, cx, selected))
                                .child(table_row::<[u16; 2]>("u16", &self.emucx, cx, selected))
                                .child(table_row::<[u8; 4]>("u8", &self.emucx, cx, selected))
                                .child(table_row::<i32>("i32", &self.emucx, cx, selected))
                                .child(table_row::<[i16; 2]>("i16", &self.emucx, cx, selected))
                                .child(table_row::<[i8; 4]>("i8", &self.emucx, cx, selected))
                                .child(table_row::<Ascii>("ascii", &self.emucx, cx, selected)),
                        ),
                ),
            )
    }
}

struct HexInputState {
    input: Entity<InputState>,
    _sub:  Subscription,
}

impl HexInputState {
    pub fn new<const PREFIX: bool, R>(
        default_value: Option<u32>,
        win: &mut Window,
        cx: &mut App,
        on_submit: impl Fn(Option<u32>, &mut App) -> R + 'static,
    ) -> Self {
        let input = cx.new(|cx| {
            let state = InputState::new(win, cx);
            match default_value {
                None => state,
                Some(default_value) => {
                    state.default_value(hex_pref::<_, PREFIX>(default_value).as_str())
                }
            }
        });

        let _sub = win.subscribe(&input, cx, move |input, event, win, cx| {
            use gpui_component::input::InputEvent;
            let (InputEvent::PressEnter { .. } | InputEvent::Blur) = event else {
                return;
            };

            let word = match parse_hex_word(&input.read(cx).value()) {
                Ok(word) => {
                    on_submit(Some(word), cx);
                    word
                }
                // TODO: handle error
                Err(_err) => {
                    on_submit(None, cx);
                    return;
                }
            };
            input.update(cx, |input, cx| {
                input.set_value(hex_pref::<_, PREFIX>(word).as_str(), win, cx)
            });
            cx.notify(input.entity_id());
        });

        HexInputState { input, _sub }
    }
}

struct MemviewTable {
    scroll:   VirtualListScrollHandle,
    emucx:    Entity<EmuContext>,
    editing:  Option<MemviewTableEdit>,
    selected: Option<u32>,
}

struct MemviewTableEdit {
    address: u32,
    input:   Entity<HexInputState>,
}

impl MemviewTable {
    fn start_edit(
        &mut self,
        address: u32,
        default_value: u32,
        win: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let emucx = self.emucx.clone();
        let view = cx.entity();
        let input = cx.new(|cx| {
            HexInputState::new::<false, _>(Some(default_value), win, cx, move |value, cx| {
                emucx.update(cx, |emucx, _| -> Option<()> {
                    let _ = emucx.emu.try_write::<u32>(address, value?);
                    None
                });
                view.update(cx, |view, _| {
                    view.editing = None;
                })
            })
        });
        input.update(cx, |input, cx| {
            cx.focus_view(&input.input, win);
        });
        self.editing = Some(MemviewTableEdit { address, input });
        self.selected = Some(address);
        cx.emit(MemviewTableEditEvent { address });
        dbg!(self.selected);
    }
}

struct MemviewTableEditEvent {
    address: u32,
}

impl EventEmitter<MemviewTableEditEvent> for MemviewTable {}

impl Render for MemviewTable {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let columns = 4u64;
        let items = u64::from(u32::MAX) / (columns * 4);
        let view = cx.entity();
        let theme = cx.theme().clone();
        VirtualList::new("memview-table", items, move |row_idx, _, cx| {
            let caddress = hex(row_idx as u32 * columns as u32 * 4);

            let mut result = h_flex()
                .font_family(&theme.mono_font_family)
                .whitespace_nowrap()
                .overflow_hidden()
                .child(
                    sel_text_keyed(
                        ElementId::NamedInteger("mem-view-row-address".into(), row_idx),
                        caddress.as_str(),
                    )
                    .text_color(theme.muted_foreground)
                    .mx_2(),
                )
                .h_6();

            for word_idx in 0..columns {
                let address = row_idx * columns * 4 + word_idx * 4;
                let address = address as u32;
                let word = view
                    .read(cx)
                    .emucx
                    .read(cx)
                    .emu
                    .try_read_pure::<u32>(address)
                    .unwrap_or(0);
                // le bytes reversed
                // let color_runs = word.to_be_bytes().map(|byte| TextRun {
                //     len: 2,
                //     font: font(&theme.mono_font_family),
                //     color: if byte == 0 {
                //         theme.muted_foreground
                //     } else {
                //         theme.foreground
                //     },
                //     ..Default::default()
                // });
                let word_str = hex_pref::<_, false>(word);

                let id = ElementId::NamedInteger("memview-hex-input".into(), u64::from(address));

                match &view.read(cx).editing.as_ref() {
                    Some(edit) if edit.address == address => {
                        result = result.child(
                            deferred(
                                Input::new(&edit.input.read(cx).input)
                                    .px_0()
                                    .h_6()
                                    .text_center()
                                    .w(rems(5.)),
                            )
                            .with_priority(1),
                        );
                    }
                    None | Some(_) => {
                        result = result.child(
                            div()
                                .id(id)
                                .child(StyledText::new(word_str.as_str()))
                                .h_6()
                                .text_center()
                                .w(rems(5.))
                                .on_click({
                                    let view = view.clone();
                                    move |_, win, cx| {
                                        view.update(cx, |view, cx| {
                                            view.start_edit(address, word, win, cx);
                                        });
                                    }
                                }),
                        );
                    }
                }
            }

            for word_idx in 0..columns {
                let address = row_idx * columns * 4 + word_idx * 4;
                let address = address as u32;
                let mut word = view
                    .read(cx)
                    .emucx
                    .read(cx)
                    .emu
                    .try_read_pure::<[u8; 4]>(address)
                    .unwrap_or([b'.'; 4]);
                let mut no_ascii = true;
                for byte in &mut word {
                    match *byte {
                        ..=0x1f | 0x7f.. => {
                            *byte = b'.';
                        }
                        _ => {
                            no_ascii = false;
                        }
                    }
                }

                let word = core::str::from_utf8(&word).expect("impossible");
                let id = ElementId::NamedInteger("mewmview-hex-ascii".into(), u64::from(address));
                let color = match (view.read(cx).editing.as_ref(), no_ascii) {
                    (Some(edit), _) if edit.address == address => &theme.colors.yellow,
                    (_, false) => &theme.foreground,
                    (_, true) => &theme.muted_foreground,
                };

                result = result.child(
                    div()
                        .child(text!(word).with_id(id))
                        .text_color(*color)
                        .w(rems(2.5)),
                );
            }

            result
        })
        // .w_full()
        .h_full()
        .track_scroll(&self.scroll)
    }
}

pub struct VirtualList {
    interactivity: Interactivity,
    rows_total:    u64,
    render_row:    Box<dyn Fn(u64, &mut Window, &mut App) -> AnyElement>,
    state:         Option<Entity<VirtualListState>>,
    scroll_handle: Option<VirtualListScrollHandle>,
}

#[derive(Clone)]
pub struct VirtualListScrollHandle {
    handle: Rc<RefCell<VirtualListScrollState>>,
}

impl VirtualListScrollHandle {
    #[must_use]
    pub fn new() -> Self {
        Self {
            handle: Rc::new(RefCell::new(VirtualListScrollState::new())),
        }
    }
}

impl Default for VirtualListScrollHandle {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VirtualListScrollState {
    state:    WeakEntity<VirtualListState>,
    deferred: Option<u64>,
}

impl VirtualList {
    pub fn new<R: IntoElement>(
        id: impl Into<ElementId>,
        rows_total: u64,
        render: impl Fn(u64, &mut Window, &mut App) -> R + 'static,
    ) -> Self {
        let mut interactivity = Interactivity::new();
        let element_id = id.into();
        interactivity.element_id = Some(element_id.clone());
        VirtualList {
            rows_total,
            render_row: Box::new(move |idx, win, cx| render(idx, win, cx).into_any_element()),
            interactivity,
            state: None,
            scroll_handle: None,
        }
    }

    #[must_use]
    pub fn track_scroll(mut self, handle: &VirtualListScrollHandle) -> Self {
        self.scroll_handle = Some(handle.clone());
        self
    }

    #[track_caller]
    fn init_state(&mut self, window: &mut Window, cx: &mut App) -> Entity<VirtualListState> {
        let state = match &self.state {
            Some(state) => state.clone(),
            None => window.use_keyed_state(
                self.interactivity
                    .element_id
                    .clone()
                    .unwrap_or(ElementId::CodeLocation(*core::panic::Location::caller())),
                cx,
                |_, _| VirtualListState::default(),
            ),
        };
        if let Some(scroll) = &self.scroll_handle {
            scroll.handle.borrow_mut().state = state.clone().downgrade();
            if let Some(deferred) = scroll.handle.borrow_mut().deferred.take() {
                state.update(cx, |state, cx| {
                    state.top_row = deferred;
                    state.frac_px = 0.0;
                    cx.notify();
                });
            }
        }
        state
    }
}

impl VirtualListScrollState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state:    WeakEntity::new_invalid(),
            deferred: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ScrollStrategy {
    #[default]
    Top,
    TopOffset(i64),
}

impl VirtualListScrollHandle {
    pub fn scroll_to(&mut self, cx: &mut impl AppContext, idx: u64, strategy: ScrollStrategy) {
        let idx = match strategy {
            ScrollStrategy::Top => idx,
            ScrollStrategy::TopOffset(offset) => idx.saturating_add_signed(-offset),
        };
        if let Some(state) = self.handle.borrow().state.upgrade() {
            state.update(cx, |state, _| state.top_row = idx)
        }
        self.handle.borrow_mut().deferred = Some(idx);
    }

    pub fn scroll_idx(&self, cx: &impl AppContext) -> u64 {
        if let Some(state) = self.handle.borrow().state.upgrade() {
            cx.read_entity(&state, |s, _| s.top_row)
        } else {
            0
        }
    }
}

impl Default for VirtualListScrollState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct VirtualListState {
    children:   Vec<(AnyElement, Point<Pixels>)>,
    top_row:    u64,
    frac_px:    f32,
    row_height: Option<f32>,
}

impl Element for VirtualList {
    type RequestLayoutState = ();
    type PrepaintState = (Entity<VirtualListState>, Hitbox);

    fn id(&self) -> Option<ElementId> {
        self.interactivity.element_id.clone()
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut gpui::App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.size.width = gpui::relative(1.0).into();
        style.size.height = gpui::relative(1.0).into();
        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
        let state = self.init_state(window, cx);
        let top_row = state.read(cx).top_row;
        let frac_px = state.read(cx).frac_px;

        let row_h: f32 = state.read(cx).row_height.unwrap_or_else(|| {
            let mut first = (self.render_row)(top_row, window, cx);
            first
                .layout_as_root(
                    Size::new(
                        AvailableSpace::Definite(bounds.size.width),
                        AvailableSpace::MinContent,
                    ),
                    window,
                    cx,
                )
                .height
                .into()
        });
        state.update(cx, |state, _| {
            state.row_height = Some(row_h);
        });

        let rows_total = self.rows_total;
        self.interactivity.on_scroll_wheel({
            let state = state.clone();
            move |ev, win, cx| {
                let Some(row_height) = state.read(cx).row_height else {
                    return;
                };
                let delta: f32 = ev.delta.pixel_delta(win.line_height()).y.into();

                state.update(cx, |state, _| {
                    let mut total = f64::from(state.frac_px) - f64::from(delta);
                    if state.top_row == 0 {
                        total = total.max(0.0);
                    }
                    let rows = (total / f64::from(row_height)).floor();
                    state.frac_px = (total - rows * f64::from(row_height)) as f32;
                    match rows >= 0.0 {
                        true => {
                            state.top_row = state
                                .top_row
                                .saturating_add(rows as u64)
                                .clamp(0, rows_total.saturating_sub(1))
                        }
                        false => state.top_row = state.top_row.saturating_sub((-rows) as u64),
                    }
                });
                cx.notify(state.entity_id());
            }
        });

        let visible = (f32::from(bounds.size.height) / row_h).ceil() as u64 + 1;
        let visible = visible.min(self.rows_total.saturating_sub(top_row));

        let mut children = Vec::with_capacity(visible as usize);

        for i in 0..visible {
            let row_index = top_row + i;
            let mut el = (self.render_row)(row_index, window, cx);

            let y = bounds.origin.y + px(i as f32 * row_h - frac_px);
            let origin = Point {
                x: bounds.origin.x,
                y,
            };

            let row_bounds = Bounds {
                origin,
                size: Size {
                    width:  bounds.size.width,
                    height: row_h.into(),
                },
            };

            el.layout_as_root(row_bounds.size.into(), window, cx);
            el.prepaint_at(origin, window, cx);

            children.push((el, origin));
        }

        state.update(cx, |state, _| state.children = children);

        (state, hitbox)
    }

    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        (state, hitbox): &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        state.update(cx, |state, cx| {
            self.interactivity.paint(
                id,
                inspector_id,
                bounds,
                Some(hitbox),
                window,
                cx,
                |_, window, cx| {
                    window.paint_quad(PaintQuad {
                        bounds,
                        background: transparent_black().into(),
                        corner_radii: Corners::all(0.0.into()),
                        border_widths: Edges::all(0.0.into()),
                        border_color: transparent_black(),
                        border_style: BorderStyle::Solid,
                    });

                    window.with_content_mask(Some(gpui::ContentMask { bounds }), |window| {
                        for (el, _origin) in &mut state.children {
                            el.paint(window, cx);
                        }
                    })
                },
            );
        });
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        Some(core::panic::Location::caller())
    }
}

impl IntoElement for VirtualList {
    type Element = Self;
    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for VirtualList {
    #[doc = " Returns a reference to the style memory of this element."]
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl InteractiveElement for VirtualList {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

struct GlobalSelection {
    state: Entity<GlobalSelectionState>,
}

struct GlobalSelectionState {
    currently_selected: Option<EntityId>,
}

impl Global for GlobalSelection {}

struct SelectionChanged(Option<EntityId>);

impl EventEmitter<SelectionChanged> for GlobalSelectionState {}

impl GlobalSelection {
    pub fn init(cx: &mut App) {
        let sel = cx.new(|_| GlobalSelectionState {
            currently_selected: None,
        });
        cx.subscribe(&sel, |sel, ev, cx| {
            sel.update(cx, |sel, _| {
                sel.currently_selected = ev.0;
            })
        })
        .detach();
        let sel = GlobalSelection { state: sel };
        cx.set_global(sel);
    }
}

pub struct Selection {
    text:       SharedString,
    focus:      FocusHandle,
    anchor:     usize,
    head:       usize,
    dragging:   bool,
    global_sub: Option<Subscription>,
}

impl Selection {
    fn range(&self) -> Range<usize> {
        self.anchor.min(self.head)..self.anchor.max(self.head)
    }
}

fn index_at(layout: &TextLayout, position: gpui::Point<gpui::Pixels>) -> usize {
    match layout.index_for_position(position) {
        Ok(ix) | Err(ix) => ix,
    }
}

/// Read-only text the pointer can select and copy.
#[derive(IntoElement)]
pub struct SelectableText {
    state: Option<Entity<Selection>>,
    id:    ElementId,
    text:  SharedString,
    style: StyleRefinement,
}

impl SelectableText {
    pub fn new(state: Entity<Selection>, text: impl Into<SharedString>) -> Self {
        let entity_id = state.entity_id();
        Self {
            state: Some(state),
            id:    ElementId::View(entity_id),
            text:  text.into(),
            style: StyleRefinement::default(),
        }
    }
}

#[track_caller]
pub fn sel_text(text: impl Into<SharedString>) -> SelectableText {
    let id = ElementId::CodeLocation(*core::panic::Location::caller());
    SelectableText {
        state: None,
        id,
        text: text.into(),
        style: StyleRefinement::default(),
    }
}

pub fn sel_text_keyed(id: impl Into<ElementId>, text: impl Into<SharedString>) -> SelectableText {
    SelectableText {
        state: None,
        id:    id.into(),
        text:  text.into(),
        style: StyleRefinement::default(),
    }
}

impl Styled for SelectableText {
    #[doc = " Returns a reference to the style memory of this element."]
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl RenderOnce for SelectableText {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let state = match self.state {
            Some(state) => state,
            None => window.use_keyed_state(self.id.clone(), cx, |_, cx| Selection {
                text:       self.text.clone(),
                focus:      cx.focus_handle(),
                anchor:     0,
                head:       0,
                dragging:   false,
                global_sub: None,
            }),
        };
        if state.read(cx).text != self.text {
            state.update(cx, |selection, _| {
                selection.text = self.text.clone();
                selection.anchor = 0;
                selection.head = 0;
            });
        }
        let (focus, range) = {
            let selection = state.read(cx);
            (selection.focus.clone(), selection.range())
        };
        let highlight = HighlightStyle {
            background_color: Some(cx.theme().colors.selection),
            ..Default::default()
        };
        let styled = StyledText::new(self.text.clone())
            .with_highlights((!range.is_empty()).then_some((range, highlight)));
        let layout = styled.layout().clone();
        let (down, drag, up, keys) = (state.clone(), state.clone(), state.clone(), state.clone());
        let (down_layout, drag_layout) = (layout.clone(), layout);
        let text = self.text;

        let global_sel = GlobalSelection::global(cx).state.clone();
        let eid = state.entity_id();
        state.update(cx, |state, cx| {
            if state.global_sub.is_none() {
                let sub = cx.subscribe(&global_sel, {
                    move |state, _, ev, _| {
                        if ev.0 != Some(eid) {
                            state.head = state.anchor;
                        }
                    }
                });
                state.global_sub = Some(sub);
            }
        });

        div()
            .id(self.id)
            .track_focus(&focus)
            .cursor(CursorStyle::IBeam)
            .refine_style(&self.style)
            .on_mouse_down(MouseButton::Left, {
                let global_sel = global_sel.clone();
                move |event, window, cx| {
                    let ix = index_at(&down_layout, event.position);
                    down.update(cx, |selection, cx| {
                        window.focus(&selection.focus, cx);
                        if !event.modifiers.shift {
                            selection.anchor = ix;
                        }
                        selection.head = ix;
                        selection.dragging = true;
                        cx.notify();
                    });

                    cx.emit(&global_sel, SelectionChanged(Some(eid)));
                }
            })
            .on_mouse_move(move |event, _, cx| {
                drag.update(cx, |selection, cx| {
                    if selection.dragging && event.dragging() {
                        selection.head = index_at(&drag_layout, event.position);
                        cx.notify();
                    }
                })
            })
            .on_mouse_up(MouseButton::Left, {
                let up = up.clone();
                move |_, _, cx| {
                    up.update(cx, |selection, _| selection.dragging = false);
                    cx.emit(&global_sel, SelectionChanged(Some(eid)));
                }
            })
            .on_mouse_up_out(MouseButton::Left, move |_, _, cx| {
                up.update(cx, |selection, _| selection.dragging = false)
            })
            .on_key_down(move |event, _, cx| {
                let stroke = &event.keystroke;
                if !stroke.modifiers.secondary() {
                    return;
                }
                match stroke.key.as_str() {
                    "c" => {
                        let range = keys.read(cx).range();
                        if !range.is_empty() {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                text[range].to_string(),
                            ));
                        }
                    }
                    "a" => keys.update(cx, |selection, cx| {
                        selection.anchor = 0;
                        selection.head = text.len();
                        cx.notify();
                    }),
                    _ => {}
                }
            })
            .child(styled)
    }
}
