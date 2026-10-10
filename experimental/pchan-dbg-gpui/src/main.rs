#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::type_complexity)]
#![allow(recursion_depth_exceeding_limit)]
#![feature(try_blocks)]

extern crate alloc;

#[path = "game-surface.rs"]
pub mod game_surface;

use alloc::borrow::Cow;
use alloc::rc::Rc;
use alloc::sync::Arc;
use core::cell::RefCell;
use core::num::{NonZeroU16, ParseIntError};
use core::ops::{DerefMut, Range};
use core::time::Duration;
use core::{fmt, mem};
use futures_lite::AsyncReadExt;
use futures_lite::io::BufReader;
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::{Input, InputState};
use gpui_component::scroll::ScrollableElement;
use gpui_component::separator::Separator;
use gpui_component::slider::{Slider, SliderEvent, SliderState};
use gpui_component::switch::Switch;
use gpui_component::tab::TabBar;
use gpui_component::{
    ActiveTheme, IconName, Selectable, Sizable, StyledExt, Theme, h_flex, v_flex,
};
use pchan_bind::ringbuf::StaticRb;
use pchan_bind::ringbuf::traits::{Consumer, Observer, RingBuffer};
use pchan_emu::debug::{Breakpoint, BreakpointKind};
use pchan_steel::{PchanSteelErr, ScriptConn, SteelCtx, SteelExecutor};
use std::path::PathBuf;
use std::time::Instant;

use bumpalo::Bump;
use gpui::prelude::*;
use gpui::{AppContext, Render, *};
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::cpu::REG_STR;
use pchan_emu::cpu::ops::OpCode;
use pchan_emu::dynarec_v2::emitters::DecodedOp;
use pchan_macros::git_rev;
use pchan_utils::{default, hex, hex_pref, init_tracing};
use schemars::JsonSchema;
use serde::Deserialize;

actions!(
    app,
    [
        Quit,
        SoftReset,
        HardReset,
        Step,
        StepFrame,
        StepBlock,
        StepInstruction,
        ViewMIPS,
        ViewMem,
        ViewRegisters,
        ViewBreakpoints,
        LoadContent,
        UnloadContent,
        ToggleRun,
        Run,
        Pause,
    ]
);

#[expect(clippy::unsafe_derive_deserialize)]
#[derive(Clone, Action, Deserialize, JsonSchema, PartialEq, Eq)]
struct CloseWindow {
    #[serde(skip)]
    id: WindowId,
}

#[expect(clippy::unsafe_derive_deserialize)]
#[derive(Clone, Action, Deserialize, JsonSchema, PartialEq, Eq)]
struct SetSpeed {
    #[serde(skip)]
    speed: EmuSpeed,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Eq)]
enum SettingsTab {
    General,
    Audio,
}

#[expect(clippy::unsafe_derive_deserialize)]
#[derive(Clone, Action, Deserialize, JsonSchema, PartialEq, Eq)]
struct ViewSettings {
    tab: SettingsTab,
}

impl ViewSettings {
    fn new(tab: SettingsTab) -> Self {
        Self { tab }
    }
}

fn window_opts() -> WindowOptions {
    gpui::WindowOptions {
        window_decorations: Some(WindowDecorations::Client),
        window_background: WindowBackgroundAppearance::Blurred,
        titlebar: Some(TitlebarOptions {
            title:                  None,
            appears_transparent:    true,
            traffic_light_position: Some(Point {
                x: 12.0.into(),
                y: 8.0.into(),
            }),
        }),
        ..Default::default()
    }
}

trait WindowOptsExt {
    fn windowed_centered(self, size: Size<Pixels>, cx: &App) -> Self;
}

impl WindowOptsExt for WindowOptions {
    fn windowed_centered(self, size: Size<Pixels>, cx: &App) -> Self {
        WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(None, size, cx))),
            ..self
        }
    }
}

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

            let steel_ctx = SteelCtx::new();
            let steel_rx = steel_ctx.rx();
            let steel_exec = SteelExecutor::new(steel_rx.clone());

            let steel = cx.new(|_| Steel {
                rx:   steel_rx,
                exec: steel_exec,
            });

            std::thread::spawn(move || {
                steel_ctx.repl().unwrap();
                std::process::exit(0)
            });

            cx.set_app_identity("p-chan-gpui", "Pーちゃん");
            let emucx = EmuContext::new(arena, cx).unwrap();
            let (target, target_buf) = create_target(
                &emucx.renderer,
                &emucx.renderer.display_uniforms.app.lock().unwrap(),
            );
            let surface_state = cx.new(|_| SurfaceState::new(target.clone(), target_buf.clone()));
            let emucx = cx.new(|_| emucx);

            let app = cx.new(|_| PChanApp {
                mips_window:        None,
                registers_window:   None,
                mem_view_window:    None,
                breakpoints_window: None,
                settings_window:    None,
                emucx:              emucx.clone(),
            });
            let appcx = AppCx {
                emucx: emucx.clone(),
                game_surface: surface_state.clone(),
                focus_handle: cx.focus_handle(),
                app: app.clone(),
                steel,
            };

            let mips_view = cx.new(|_| MipsView {
                scroll_handle: VirtualListScrollHandle::new(),
                appcx:         appcx.clone(),
                pc:            0xbfc0_0000,
            });
            let mem_view = cx.new(|_| MemviewTable {
                scroll_handle: VirtualListScrollHandle::new(),
                appcx:         appcx.clone(),
                editing:       None,
                selected:      None,
            });
            let cpu_reg_names: [_; 32] =
                core::array::from_fn(|i| format!("${}", REG_STR[i]).into());
            let reg_view = cx.new(|_| RegView {
                appcx:                appcx.clone(),
                cached_cpu_reg_names: cpu_reg_names,
                cpu_control_reg_tab:  0,
                cpu_control_subs:     [const { None }; 32],
            });
            let breakpoints_view = cx.new(|_| BreakpointsView {
                appcx: appcx.clone(),
            });

            let game_view = cx.new(|_| GameView {
                appcx: appcx.clone(),
            });

            let settings_view = cx.new(|_| SettingsView {
                appcx: appcx.clone(),
                tab:   SettingsTab::Audio,
            });

            app.update(cx, |app, cx| {
                app.set_menus(cx);
            });

            set_subwindow::<ViewMIPS, _>(
                &app,
                &mips_view,
                cx,
                |app| &mut app.mips_window,
                |cx| window_opts().windowed_centered(size(px(400.), px(400.)), cx),
            );
            set_subwindow::<ViewMem, _>(
                &app,
                &mem_view,
                cx,
                |app| &mut app.mem_view_window,
                |cx| window_opts().windowed_centered(size(px(640.), px(400.)), cx),
            );
            set_subwindow::<ViewRegisters, _>(
                &app,
                &reg_view,
                cx,
                |app| &mut app.registers_window,
                |cx| window_opts().windowed_centered(size(px(450.), px(300.)), cx),
            );
            set_subwindow::<ViewBreakpoints, _>(
                &app,
                &breakpoints_view,
                cx,
                |app| &mut app.breakpoints_window,
                |cx| window_opts().windowed_centered(size(px(350.), px(480.)), cx),
            );
            set_subwindow::<ViewSettings, _>(
                &app,
                &settings_view,
                cx,
                |app| &mut app.settings_window,
                |cx| window_opts().windowed_centered(size(px(480.), px(480.)), cx),
            );

            cx.on_action(listener(&app, PChanApp::load_content_listener));

            // let theme_reg = ThemeRegistry::global_mut(cx);
            // let gruvbox = include_str!("./assets/themes/gruvbox.json");
            // theme_reg
            //     .load_themes_from_str(gruvbox)
            //     .expect("failed to load theme from string");
            // let gruvbox = theme_reg.themes().get("Gruvbox Dark").unwrap().clone();

            let theme = cx.global_mut::<Theme>();
            theme.apply_config(&Rc::new(gpui_component::ThemeConfig {
                mode: gpui_component::ThemeMode::Dark,
                ..default()
            }));
            theme.primary = rgb_to_hsla(rgba(0xb1e6b7ff));
            cx.set_window_appearance(Some(WindowAppearance::VibrantDark));

            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.bind_keys([KeyBinding::new("cmd-j", ToggleRun, None)]);
            cx.bind_keys([KeyBinding::new("cmd-k", StepBlock, None)]);

            cx.on_action(move |_: &Quit, cx| {
                #[cfg(feature = "dhat-heap")]
                unsafe {
                    core::ptr::drop_in_place(_profiler_ptr);
                }
                cx.quit();
            });
            cx.on_action(move |action: &ViewSettings, cx| {
                cx.propagate();
                settings_view.update(cx, |settings, cx| {
                    settings.tab = action.tab;
                    cx.notify();
                });
            });

            cx.spawn({
                async move |cx| {
                    emu_loop(appcx.clone(), cx).await;
                }
            })
            .detach();

            _ = cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                        None,
                        size(px(640.), px(480.)),
                        cx,
                    ))),
                    ..window_opts()
                },
                |win, cx| {
                    game_view.update(cx, |game_view, cx| {
                        win.focus(&game_view.appcx.focus_handle, cx);
                    });

                    game_view
                },
            );

            cx.activate(true);
        });
}

struct PChanApp {
    mips_window:        Option<Subwindow<MipsView>>,
    mem_view_window:    Option<Subwindow<MemviewTable>>,
    registers_window:   Option<Subwindow<RegView>>,
    breakpoints_window: Option<Subwindow<BreakpointsView>>,
    settings_window:    Option<Subwindow<SettingsView>>,
    emucx:              Entity<EmuContext>,
}

#[derive(Clone)]
struct AppCx {
    emucx:        Entity<EmuContext>,
    game_surface: Entity<SurfaceState>,
    app:          Entity<PChanApp>,
    steel:        Entity<Steel>,
    focus_handle: FocusHandle,
}

struct Steel {
    rx:   ScriptConn,
    exec: SteelExecutor,
}

struct GameView {
    appcx: AppCx,
}

#[derive(Copy)]
struct Subwindow<T> {
    handle: WindowHandle<T>,
    active: bool,
}

impl<T> fmt::Debug for Subwindow<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Subwindow")
            .field("handle", &self.handle)
            .field("active", &self.active)
            .finish()
    }
}

impl<T> Clone for Subwindow<T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
            active: self.active,
        }
    }
}

impl EmuContext {
    fn content_path(&self) -> Option<&PathBuf> {
        match &self.content_path {
            ContentPath::None => None,
            ContentPath::Disc(path_buf) | ContentPath::Exe(path_buf) => Some(path_buf),
        }
    }

    fn content_title(&self) -> Option<String> {
        match self.content_path() {
            Some(path) => path.file_name().map(|f| f.to_string_lossy().into_owned()),
            None => None,
        }
    }
}

impl PChanApp {
    pub fn set_menus(&self, cx: &mut impl DerefMut<Target = App>) {
        let menus = self.menu(cx);
        cx.defer(move |cx| {
            cx.set_menus(menus);
        });
    }

    pub fn menu(&self, cx: &App) -> impl IntoIterator<Item = Menu> + use<> {
        fn speed_menu_item(speed: EmuSpeed) -> MenuItem {
            let name = match speed {
                EmuSpeed::Unlimited => "Unlimited".to_string(),
                EmuSpeed::Percentage(non_zero) => format!("{}%", non_zero.get()),
            };
            MenuItem::action(name, SetSpeed { speed })
        }
        [
            Menu::new("P-chan").items([
                MenuItem::action("💿 Load Content", LoadContent),
                MenuItem::action("Unload content", UnloadContent)
                    .disabled(self.emucx.read(cx).content_path().is_none()),
                MenuItem::separator(),
                MenuItem::action("Run", Run),
                MenuItem::action("Pause", Pause),
                MenuItem::submenu(Menu::new("Reset").items([
                    MenuItem::action("Hard reset", HardReset),
                    MenuItem::action("Soft reset", SoftReset),
                ])),
                MenuItem::separator(),
                MenuItem::SystemMenu(OsMenu {
                    name:      "Services".into(),
                    menu_type: SystemMenuType::Services,
                }),
            ]),
            Menu::new("Settings").items([
                MenuItem::submenu(
                    Menu::new("Speed").items(
                        [
                            EmuSpeed::percentage(100),
                            EmuSpeed::percentage(150),
                            EmuSpeed::percentage(200),
                            EmuSpeed::percentage(300),
                            EmuSpeed::Unlimited,
                        ]
                        .map(|speed| {
                            speed_menu_item(speed)
                                .checked(speed == self.emucx.read(cx).runner.config.speed)
                        }),
                    ),
                ),
                MenuItem::separator(),
                MenuItem::action("Audio Settings", ViewSettings::new(SettingsTab::Audio)),
            ]),
            Menu::new("Debug").items([
                MenuItem::action("Run", Run),
                MenuItem::action("Pause", Pause),
                MenuItem::submenu(Menu::new("Reset").items([
                    MenuItem::action("Hard reset", HardReset),
                    MenuItem::action("Soft reset", SoftReset),
                ])),
                MenuItem::separator(),
                MenuItem::action("Step Block", StepBlock),
                MenuItem::action("Step Instruction", StepInstruction),
                MenuItem::action("Step Frame", StepFrame),
                MenuItem::separator(),
                MenuItem::action("MIPS Dump", ViewMIPS)
                    .checked(Self::view_active(self.mips_window.as_ref())),
                MenuItem::action("CPU Registers", ViewRegisters)
                    .checked(Self::view_active(self.registers_window.as_ref())),
                MenuItem::separator(),
                MenuItem::action("Memory", ViewMem)
                    .checked(Self::view_active(self.mem_view_window.as_ref())),
                MenuItem::action("Breakpoints", ViewBreakpoints)
                    .checked(Self::view_active(self.registers_window.as_ref())),
            ]),
        ]
    }

    fn view_active<T>(win: Option<&Subwindow<T>>) -> bool {
        let Some(win) = win else {
            return false;
        };
        win.active
    }

    fn load_content(&mut self, cx: &mut Context<Self>) -> Task<miette::Result<()>> {
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
                            let fsm = emucx.emu.open_disc(&content_path, true).into_diagnostic()?;
                            emucx
                                .emu
                                .advance_open_disc(&content_path, fsm, true)
                                .into_diagnostic()?;
                            emucx.content_path = ContentPath::Disc(content_path);
                            Ok(())
                        })?;
                        view.set_menus(cx);
                        cx.notify();
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
                        view.emucx.update(cx, |emucx, cx| -> miette::Result<()> {
                            emucx.emu.sideload_exe(&exe).into_diagnostic()?;
                            emucx.content_path = ContentPath::Disc(content_path);
                            Ok(())
                        })?;
                        view.set_menus(cx);
                        cx.notify();
                        Ok(())
                    })?;
                }
            }
            Ok(())
        })
    }

    fn load_content_listener(&mut self, _: &LoadContent, cx: &mut Context<Self>) {
        self.load_content(cx).detach();
    }
}

async fn emu_loop(appcx: AppCx, cx: &mut AsyncApp) {
    let emucx = appcx.emucx;
    let surface = appcx.game_surface;
    let steel_conn = cx.read_entity(&appcx.steel, |steel, _| steel.rx.clone());
    let mut time_since_last_yield = Duration::ZERO;
    loop {
        let run_listener = emucx.update(cx, |emucx, _| match emucx.runner.running {
            false => Some(emucx.running_notify.listen()),
            true => None,
        });

        if let Some(run_listener) = run_listener {
            let signal = futures_lite::future::race(
                async {
                    run_listener.await;
                    None
                },
                async {
                    let msg = steel_conn.chan.1.recv().await.unwrap();
                    Some(msg)
                },
            );
            if let Some(msg) = signal.await {
                let summary = appcx.steel.update(cx, |steel, cx| {
                    emucx.update(cx, |emucx, _| {
                        steel
                            .exec
                            .handle_call(&mut emucx.emu, &mut emucx.runner, msg)
                    })
                });

                if let Some(tx) = summary.open_content {
                    let task = appcx.app.update(cx, |app, cx| app.load_content(cx));
                    cx.background_spawn(async move {
                        let res = task.await.map_err(PchanSteelErr::OpenContentError);
                        tx.as_async().send(res).await.unwrap();
                    })
                    .detach();
                }
            }
            emucx.update(cx, |emucx, _| {
                emucx.start = Instant::now();
                emucx.cycles_per_run = 0;
                emucx.real_time_running = Duration::ZERO;
            });
        } else {
            appcx.steel.update(cx, |steel, cx| {
                let Ok(Some(msg)) = steel.rx.chan.1.try_recv() else {
                    return;
                };
                emucx.update(cx, |emucx, _| {
                    steel
                        .exec
                        .handle_call(&mut emucx.emu, &mut emucx.runner, msg);
                });
            });
        }

        let deadline = emucx.update(cx, |emucx, cx| {
            surface.as_mut(cx).start_display_draw(&emucx.renderer);

            let elapsed = emucx.runner.run_until_vblank_with(&mut emucx.emu, |emu| {
                emucx.pc_history.push_overwrite(emu.cpu.pc);
                emucx.run_once
            });

            appcx.steel.update(cx, |steel, _| {
                steel.exec.handle_step(&mut emucx.emu, &mut emucx.runner);
            });

            if emucx.run_for_one_frame {
                emucx.runner.running = false;
                emucx.run_for_one_frame = false;
                cx.notify();
            }
            if emucx.run_once {
                emucx.run_once = false;
                emucx.runner.running = false;
                cx.notify();
            }

            let deadline = if let Some(elapsed) = elapsed {
                let sleep_time = emucx.runner.sleep_time(elapsed);
                emucx.frame_time = elapsed;
                emucx.frame_time_limited = elapsed.max(emucx.runner.frame_time_limit());
                emucx.frame_times.push_overwrite(elapsed.as_millis() as u16);
                time_since_last_yield += elapsed;
                Instant::now() + sleep_time
            } else {
                Instant::now()
            };

            let surface = surface.as_mut(cx);
            surface.wait_for_display_draw(&emucx.renderer);

            deadline
        });

        if !spin_sleep2(cx, deadline).await
            && time_since_last_yield > Duration::from_micros(16_667 * 2)
        {
            futures_lite::future::yield_now().await;
            time_since_last_yield = Duration::ZERO;
        }
    }
}

impl PchanAppActions for GameView {
    fn appcx(&self) -> &AppCx {
        &self.appcx
    }
}

impl Render for GameView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let surface = GameSurface::new(
            "game-surface",
            cx.read_entity(&self.appcx.emucx, |emucx, _| emucx.renderer.clone()),
            self.appcx.game_surface.clone(),
        )
        .w_full()
        .h_full()
        .min_h_4()
        .min_w_4();

        let content = self.appcx.emucx.read(cx).content_title();
        let view = v_flex()
            .h_full()
            .w_full()
            .justify_end()
            .pchan_actions(self, cx)
            .child(div().w_full().flex_grow_1().child(surface))
            .child({
                let theme = cx.theme();
                let frame_idx = self.appcx.emucx.read(cx).runner.frame_idx;
                let frame_time_limited = self.appcx.emucx.read(cx).frame_time_limited;
                let frame_time_limited_secs = frame_time_limited.as_secs_f32();
                let frame_time_limited_ms = frame_time_limited.as_millis();
                let frame_time_ms = self
                    .appcx
                    .emucx
                    .read(cx)
                    .frame_times
                    .iter()
                    .copied()
                    .fold(0u64, |acc, ms| acc + u64::from(ms))
                    / self.appcx.emucx.read(cx).frame_times.occupied_len().max(1) as u64;

                let _running = self.appcx.emucx.read(cx).runner.running;
                let fps = 1.0 / frame_time_limited_secs;
                let speed = 16 * 100 / (frame_time_limited_ms.max(1));
                let cpu_mode = self.appcx.emucx.read(cx).runner.mode();
                let resolution = self.appcx.emucx.read(cx).emu.gpu.gpustat.resolution();
                let video_mode = self.appcx.emucx.read(cx).emu.gpu.gpustat.video_mode();

                h_flex()
                    .text_sm()
                    .text_color(theme.foreground)
                    .bg(theme.background)
                    .border_t_1()
                    .border_color(theme.border)
                    .justify_end()
                    .whitespace_nowrap()
                    .flex_nowrap()
                    .overflow_hidden()
                    .px_4()
                    .gap_4()
                    .child(
                        h_flex()
                            .flex_grow_1()
                            .child(sel_text(format!("#{frame_idx}"))),
                    )
                    .child(Separator::vertical().h_full())
                    .child(div().child(format!("{cpu_mode:?}")))
                    .child(Separator::vertical().h_full())
                    .child(div().child(format!("{}x{} {}", resolution.x, resolution.y, video_mode)))
                    .when(frame_time_limited_secs != 0.0, |this| {
                        this.child(Separator::vertical().h_full())
                            .child(div().w(rems(6.5)).child(format!("{fps:.0} FPS ({speed}%)")))
                    })
                    .when(frame_time_ms != 0, |this| {
                        this.child(Separator::vertical().h_full())
                            .child(div().w(rems(2.)).child(format!("{frame_time_ms}ms")))
                    })
            });

        view.wrap_window(cx).title(None).content(content)
    }
}

fn listener<T, A, R, U>(entity: &Entity<T>, func: U) -> impl Fn(&A, &mut App) -> R + use<T, A, R, U>
where
    T: 'static,
    U: Fn(&mut T, &A, &mut Context<T>) -> R,
{
    let e = entity.clone();
    move |action: &A, cx: &mut App| e.update(cx, |e, cx| func(e, action, cx))
}

trait PchanAppActions: Sized + 'static {
    fn appcx(&self) -> &AppCx;

    fn run(&mut self, _: &Run, _: &mut Window, cx: &mut Context<Self>) {
        self.appcx().emucx.update(cx, |emucx, _| {
            emucx.runner.running = true;
            emucx.running_notify.notify(usize::MAX);
        });
        cx.notify();
    }

    fn pause<T: 'static>(&mut self, _: &Pause, _: &mut Window, cx: &mut Context<T>) {
        self.appcx().emucx.update(cx, |emucx, _| {
            emucx.runner.running = false;
        });
        cx.notify();
    }

    fn toggle_run(&mut self, _: &ToggleRun, _: &mut Window, cx: &mut Context<Self>) {
        self.appcx().emucx.update(cx, |emucx, cx| {
            if emucx.runner.running {
                emucx.runner.running = false;
            } else {
                emucx.runner.running = true;
                emucx.running_notify.notify(usize::MAX);
            }
            cx.notify();
        });
        cx.notify();
    }

    fn hard_reset(&mut self, _: &HardReset, _: &mut Window, cx: &mut Context<Self>) {
        self.appcx().emucx.update(cx, |emucx, cx| {
            emucx.runner.frame_idx = 0;

            emucx.emu.gpu.wait_for_render_result();
            let bios_path = emucx.emu.boot.bios_path.clone();
            let alloc = emucx.emu.alloc;
            emucx.emu.gpu.reset_renderer();
            let mut new = Emu::new_in(alloc);
            new.set_bios_path(bios_path);
            new.load_bios(alloc).unwrap();
            new.cpu.jump_to_bios();
            new.tty.set_tracing();

            mem::swap(&mut new.gpu.conn, &mut emucx.emu.gpu.conn);
            new.spu.put_prod(emucx.emu.spu.take_prod());

            emucx.emu = new;
        });
        cx.notify();
    }

    fn unload_content(&mut self, _: &UnloadContent, _: &mut Window, _: &mut Context<Self>) {}

    fn set_speed(&mut self, speed: &SetSpeed, _: &mut Window, cx: &mut Context<Self>) {
        self.appcx().emucx.update(cx, |emucx, _| {
            emucx.runner.config.speed = speed.speed;
        });
        self.appcx().app.update(cx, |app, cx| {
            app.set_menus(cx);
        });
        cx.notify();
    }

    fn set_run_once(&mut self, cx: &mut Context<Self>) {
        self.appcx().emucx.update(cx, |emucx, _| {
            emucx.runner.running = true;
            emucx.run_once = true;
            emucx.running_notify.notify(usize::MAX);
        });
    }

    fn step_block(&mut self, _: &StepBlock, _: &mut Window, cx: &mut Context<Self>) {
        self.set_run_once(cx);
    }
    fn step_instruction(&mut self, _: &StepInstruction, _: &mut Window, cx: &mut Context<Self>) {
        self.set_run_once(cx);
    }
    fn step_frame(&mut self, _: &StepFrame, _: &mut Window, cx: &mut Context<Self>) {
        self.appcx().emucx.update(cx, |emucx, _| {
            emucx.runner.running = true;
            emucx.run_for_one_frame = true;
            emucx.running_notify.notify(usize::MAX);
        });
    }
}

trait RegisterPchanActions {
    fn pchan_actions<T: PchanAppActions>(self, view: &T, cx: &mut Context<T>) -> Self;
}

impl<E> RegisterPchanActions for E
where
    E: IntoElement + gpui::InteractiveElement,
{
    fn pchan_actions<T: PchanAppActions>(self, view: &T, cx: &mut Context<T>) -> Self {
        let running = view.appcx().emucx.read(cx).runner.running;

        self.when(running, |this| this.on_action(cx.listener(T::pause)))
            .when(!running, |this| this.on_action(cx.listener(T::run)))
            .when(
                view.appcx().emucx.read(cx).content_path().is_some(),
                |this| this.on_action(cx.listener(T::unload_content)),
            )
            .map(|this| match view.appcx().emucx.read(cx).runner.mode() {
                RunnerMode::Dynarec => this.on_action(cx.listener(T::step_block)),
                RunnerMode::Interpreter => this.on_action(cx.listener(T::step_instruction)),
            })
            .on_action(cx.listener(T::step_frame))
            .on_action(cx.listener(T::hard_reset))
            .on_action(cx.listener(T::set_speed))
            .on_action(cx.listener(T::toggle_run))
            .track_focus(&view.appcx().focus_handle)
    }
}

fn set_subwindow<T: Action, V: Render + PchanAppActions>(
    app: &Entity<PChanApp>,
    view: &Entity<V>,
    cx: &mut App,
    get_field: for<'a> fn(&'a mut PChanApp) -> &'a mut Option<Subwindow<V>>,
    window_options: impl Fn(&App) -> WindowOptions + 'static,
) {
    cx.on_action({
        let app = app.clone();
        move |close: &CloseWindow, cx| {
            cx.propagate();
            let app = app.clone();
            let close = close.clone();
            cx.defer(move |cx| {
                let mut window = app.update(cx, |app, _| get_field(app).clone());
                if let Some(sw) = &mut window {
                    _ = sw
                        .handle
                        .update(cx, |_, win, _| {
                            if win.window_handle().window_id() == close.id {
                                win.minimize_window();
                                sw.active = false;
                            }
                        })
                        .inspect_err(|err| eprintln!("{err}"));
                }
                app.update(cx, |app, cx| {
                    *get_field(app) = window;
                    app.set_menus(cx);
                });
            });
        }
    });
    let window_options = Arc::new(window_options);
    cx.on_action({
        let app = app.clone();
        let view = view.clone();
        move |_: &T, cx| {
            let app_view = app.clone();
            let view = view.clone();
            let focus_handle = view.read(cx).appcx().focus_handle.clone();
            let window_options = window_options.clone();
            cx.propagate();
            cx.defer(move |cx| {
                let window = app_view.update(cx, |app, _| get_field(app).clone());
                let active = match window {
                    Some(win) => win
                        .handle
                        .update(cx, |_, win, cx| {
                            if win.is_window_active() {
                                win.minimize_window();
                                false
                            } else {
                                win.activate_window();
                                win.focus(&focus_handle, cx);
                                true
                            }
                        })
                        .map_err(|err| miette!("{err}"))
                        .unwrap(),
                    None => {
                        let win = cx
                            .open_window(window_options(cx), {
                                let app_view = app_view.clone();
                                move |win, cx| {
                                    cx.bind_keys([KeyBinding::new(
                                        "cmd-w",
                                        CloseWindow {
                                            id: win.window_handle().window_id(),
                                        },
                                        None,
                                    )]);
                                    win.on_window_should_close(cx, move |_, cx| {
                                        app_view.update(cx, |app, cx| {
                                            *get_field(app) = None;
                                            app.set_menus(cx);
                                        });
                                        true
                                    });
                                    win.focus(&focus_handle, cx);
                                    view
                                }
                            })
                            .map_err(|err| miette!("{err}"))
                            .unwrap();

                        app_view.update(cx, |app, _| {
                            *get_field(app) = Some(Subwindow {
                                handle: win,
                                active: true,
                            });
                        });

                        true
                    }
                };

                app_view.update(cx, |app, cx| {
                    if let Some(sw) = get_field(app) {
                        sw.active = active
                    }
                    app.set_menus(cx);
                })
            });
        }
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
            _ => panic!("no asset named {path}"),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

struct MipsView {
    scroll_handle: VirtualListScrollHandle,
    appcx:         AppCx,
    pc:            u32,
}

impl PchanAppActions for MipsView {
    fn appcx(&self) -> &AppCx {
        &self.appcx
    }
}

impl Render for MipsView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        let theme = cx.theme().clone();

        let pc = self.appcx().emucx.read(cx).emu.cpu.pc;

        if pc != self.pc {
            self.scroll_handle
                .scroll_to(cx, u64::from(pc) / 4, ScrollStrategy::TopOffset(4));
            self.pc = pc;
        }

        let content = v_flex()
            .w_full()
            .text_color(theme.foreground)
            .h_full()
            .bg(transparent_black())
            .child(
                sel_text(format!("$pc: {}", hex(pc)))
                    .bg(theme.background.opacity(0.9))
                    .font_family(&theme.mono_font_family),
            )
            .child(
                VirtualList::new(
                    "mips-dump-list",
                    u64::from(u32::MAX) / 4,
                    move |idx, _, cx| {
                        let address_label: SharedString = "mips-dump-address".into();

                        let view = view.read(cx);
                        let addr = idx as u32 * 4;
                        let instr = view.appcx.emucx.read(cx).emu.fastmem_read::<OpCode>(addr);
                        let instr = instr
                            .map(DecodedOp::new)
                            .map_or(Cow::Borrowed("N/A"), |instr| Cow::Owned(format!("{instr}")));
                        let is_pc = pc & 0x1fff_ffff == addr & 0x1fff_ffff;
                        let mut color = cx.theme().background.opacity(0.9);
                        if idx.is_multiple_of(2) {
                            color.lightness += 0.05;
                        }
                        h_flex()
                            .w_full()
                            .whitespace_nowrap()
                            .overflow_hidden()
                            .font_family(&theme.mono_font_family)
                            .bg(color)
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
                            .when(is_pc, |this| this.text_color(theme.primary))
                            .h_4()
                    },
                )
                .track_scroll(&self.scroll_handle),
            );

        content
            .pchan_actions(self, cx)
            .wrap_window(cx)
            .title(Some("MIPS Dump"))
            .content(self.appcx.emucx.read(cx).content_title())
            .into_element()
            .bg(transparent_black())
    }
}

enum ContentPath {
    None,
    Disc(PathBuf),
    Exe(PathBuf),
}

pub struct EmuContext {
    emu:                Emu<&'static Bump>,
    runner:             Runner<&'static Bump>,
    run_for_one_frame:  bool,
    run_once:           bool,
    running_notify:     event_listener::Event,
    renderer:           Arc<pchan_gpu::Renderer>,
    frame_time:         Duration,
    frame_time_limited: Duration,
    frame_times:        StaticRb<u16, 10>,
    real_time_running:  Duration,
    cycles_per_run:     u64,
    start:              Instant,
    pc_history:         StaticRb<u32, 50>,
    content_path:       ContentPath,
    alloc:              &'static Bump,
}

struct RenderedFrameEvent;
struct ResetEvent;

impl EventEmitter<RenderedFrameEvent> for EmuContext {}
impl EventEmitter<ResetEvent> for EmuContext {}

impl EmuContext {
    pub fn runner_mode(&self) -> RunnerMode {
        self.runner.mode()
    }
}

use miette::{IntoDiagnostic, miette};
use pchan_emu::run::{EmuSpeed, Runner, RunnerMode};

use crate::game_surface::{GameSurface, SurfaceState, create_target};

impl EmuContext {
    pub fn new(alloc: &'static Bump, cx: &App) -> miette::Result<Self> {
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

        let mut runner = Runner::new_in(alloc).with_config(pchan_emu::run::RunnerConfig {
            force_mode: Some(pchan_emu::run::RunnerMode::Dynarec),
            speed:      pchan_emu::run::EmuSpeed::Percentage(NonZeroU16::new(100).unwrap()),
        });
        runner.running = false;

        Ok(EmuContext {
            emu,
            renderer: gpu.clone(),
            running_notify: event_listener::Event::new(),
            run_once: false,
            runner,
            frame_time: Duration::ZERO,
            frame_time_limited: Duration::ZERO,
            frame_times: StaticRb::default(),
            start: Instant::now(),
            cycles_per_run: 0,
            real_time_running: Duration::ZERO,
            run_for_one_frame: false,
            alloc,
            pc_history: StaticRb::default(),
            content_path: ContentPath::None,
        })
    }
}

async fn spin_sleep2(cx: &AsyncApp, deadline: Instant) -> bool {
    let mut yielded = false;
    let sleep_for = deadline
        .saturating_duration_since(Instant::now())
        .saturating_sub(Duration::from_millis(3));
    if sleep_for > Duration::ZERO {
        yielded = true;
        cx.background_executor()
            .spawn_with_priority(Priority::High, cx.background_executor().timer(sleep_for))
            .await;
    }
    loop {
        if Instant::now() > deadline {
            return yielded;
        }
        core::hint::spin_loop();
    }
}

fn titlebar(title: Option<&str>, content: Option<String>, theme: &Theme) -> impl IntoElement {
    use core::fmt::Write;

    let mut title_text: String = "🐷🎗️ P-ちゃん".into();
    _ = match title {
        Some(title) => write!(title_text, " | {title}"),
        None => write!(title_text, " ({})", git_rev!()),
    };
    if let Some(content) = content {
        _ = write!(title_text, " | {content}");
    }

    h_flex()
        .text_sm()
        .bg(theme.background)
        .py_1()
        .gap_2()
        .items_center()
        .justify_center()
        .font_weight(FontWeight::BOLD)
        .w_full()
        .text_color(theme.foreground)
        .relative()
        .child(title_text)
        .child(
            div()
                .bg(theme.background)
                .absolute()
                .top_0()
                .left_0()
                .w(rems(5.))
                .h_full(),
        )
}

fn window_wrapper<U: IntoElement>(
    title: Option<&str>,
    content: Option<String>,
    theme: &Theme,
    view: U,
) -> Div {
    v_flex()
        .size_full()
        .child(titlebar(title, content, theme))
        .backdrop_blur(8.0)
        .bg(theme.background.opacity(0.9))
        .text_color(theme.foreground)
        .child(div().w_full().h_full().child(view))
}

struct WindowWrapper<'a, T> {
    title:   Option<&'a str>,
    content: Option<String>,
    theme:   Theme,
    view:    T,
}

impl<T: IntoElement> IntoElement for WindowWrapper<'_, T> {
    type Element = <Div as IntoElement>::Element;

    fn into_element(self) -> Self::Element {
        window_wrapper(self.title, self.content, &self.theme, self.view).into_element()
    }
}

trait WrapWindow: Sized {
    fn wrap_window<'a>(self, cx: &App) -> WindowWrapper<'a, Self>;
}

impl<T> WrapWindow for T
where
    T: IntoElement,
{
    fn wrap_window<'a>(self, cx: &App) -> WindowWrapper<'a, Self> {
        WindowWrapper {
            title:   None,
            content: None,
            theme:   cx.theme().clone(),
            view:    self,
        }
    }
}

impl<'a, T> WindowWrapper<'a, T> {
    fn title(mut self, title: Option<&'a str>) -> Self {
        self.title = title;
        self
    }

    fn content(mut self, content: Option<String>) -> Self {
        self.content = content;
        self
    }
}

struct BreakpointsView {
    appcx: AppCx,
}

impl PchanAppActions for BreakpointsView {
    fn appcx(&self) -> &AppCx {
        &self.appcx
    }
}

impl Render for BreakpointsView {
    fn render(&mut self, win: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        fn bp_toggle(
            id: impl Into<ElementId>,
            address: u32,
            kind: BreakpointKind,
            bp: Breakpoint,
            label: &'static str,
            cx: &Context<BreakpointsView>,
        ) -> Switch {
            Switch::new(id)
                .small()
                .checked(bp.kind.contains(kind))
                .label(label)
                .flex_grow_0()
                .min_h_0()
                .on_click(cx.listener(move |view, ev, _, cx| {
                    view.appcx.emucx.update(cx, |emucx, _| {
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

        let emucx = self.appcx.emucx.clone();
        let theme = cx.theme().clone();

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
        let breakpoints =
            self.appcx
                .emucx
                .read(cx)
                .emu
                .dbg
                .breakpoints
                .iter()
                .map(|(address, bp)| {
                    let address = *address;
                    let idn = u64::from(address);

                    h_flex()
                        .gap_2()
                        .text_sm()
                        .min_h_0()
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
                                .ghost()
                                .small()
                                .aspect_square()
                                .cursor_pointer()
                                .label("X")
                                .on_click(cx.listener(move |state, _, _, cx| {
                                    state.appcx.emucx.update(cx, |emucx, _| {
                                        emucx.emu.dbg.remove_breakpoint(address);
                                    })
                                })),
                        )
                });

        v_flex()
            .size_full()
            .gap_2()
            .child(
                v_flex()
                    .gap_2()
                    .p_2()
                    .w_full()
                    .h_1_2()
                    .child(Input::new(&add_bp_input))
                    .child(
                        div()
                            .v_flex()
                            .size_full()
                            .overflow_y_scrollbar()
                            .min_h_0()
                            .children(breakpoints),
                    ),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(Separator::horizontal().flex_grow_1())
                    .child(div().child("History").text_color(theme.muted_foreground))
                    .child(Separator::horizontal().flex_grow_1()),
            )
            .child(
                v_flex()
                    .w_full()
                    .h_1_2()
                    .overflow_y_scrollbar()
                    .min_h_0()
                    .p_2()
                    .children(
                        self.appcx
                            .emucx
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
            .pchan_actions(self, cx)
            .wrap_window(cx)
            .title(Some("Breakpoints"))
            .content(self.appcx.emucx.read(cx).content_title())
    }
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

struct RegView {
    appcx:                AppCx,
    cached_cpu_reg_names: [SharedString; 32],
    cpu_control_reg_tab:  usize,
    cpu_control_subs:     [Option<Subscription>; 32],
}

impl PchanAppActions for RegView {
    fn appcx(&self) -> &AppCx {
        &self.appcx
    }
}

impl Render for RegView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();

        let tabbar = TabBar::new("segmented-tabs")
            .min_h_0()
            .segmented()
            .selected_index(self.cpu_control_reg_tab)
            .cursor_pointer()
            .on_click(cx.listener(|view, index, _, cx| {
                view.cpu_control_reg_tab = *index;
                cx.notify();
            }))
            .children(vec!["CPU", "COP0", "GTE"]);

        let gpr = self.appcx.emucx.read(cx).emu.cpu.gpr.clone();
        let gpr = gpr.iter().copied().enumerate().map({
            |(r, value)| {
                use gpui_component::input::InputEvent;

                let reg_id = &self.cached_cpu_reg_names[r];
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
                                                view.appcx.emucx.update(cx, |emucx, _| {
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
                    0 => theme.muted_foreground,
                    _ => theme.foreground,
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
                    .child(
                        div()
                            .text_color(theme.foreground)
                            .child(reg_id.clone())
                            .text_ellipsis()
                            .w(rems(2.)),
                    )
                    .child(
                        Input::new(&input_state)
                            .appearance(input_state.focus_handle(cx).is_focused(window))
                            .w(rems(9.))
                            .text_color(color)
                            .flex_grow_0(),
                    )
            }
        });
        v_flex()
            .id("cpu-scroll-container")
            .gap_2()
            .min_w_0()
            .min_h_0()
            .size_full()
            .p_2()
            .child(tabbar)
            .child(
                div()
                    .gap_1()
                    .v_flex()
                    .flex_wrap()
                    .min_w_0()
                    .min_h_0()
                    .overflow_x_scrollbar()
                    .children(gpr),
            )
            .pchan_actions(self, cx)
            .wrap_window(cx)
            .title(Some("CPU Reg."))
            .content(self.appcx.emucx.read(cx).content_title())
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
        on_submit: impl Fn(Option<u32>, &mut Window, &mut App) -> R + 'static,
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
                    on_submit(Some(word), win, cx);
                    word
                }
                // TODO: handle error
                Err(_err) => {
                    on_submit(None, win, cx);
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
    scroll_handle: VirtualListScrollHandle,
    appcx:         AppCx,
    editing:       Option<MemviewTableEdit>,
    selected:      Option<u32>,
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
        let emucx = self.appcx().emucx.clone();
        let view = cx.entity();
        let own_focus = self.appcx().focus_handle.clone();
        let input = cx.new(|cx| {
            HexInputState::new::<false, _>(Some(default_value), win, cx, move |value, win, cx| {
                emucx.update(cx, |emucx, _| -> Option<()> {
                    let _ = emucx.emu.try_write::<u32>(address, value?);
                    None
                });
                view.update(cx, |view, cx| {
                    view.editing = None;
                    win.focus(&own_focus, cx);
                    cx.notify();
                })
            })
        });
        input.update(cx, |input, cx| {
            cx.focus_view(&input.input, win);
        });
        self.editing = Some(MemviewTableEdit { address, input });
        self.selected = Some(address);
        cx.emit(MemviewTableEditEvent { address });
    }
}

struct MemviewTableEditEvent {
    address: u32,
}

impl EventEmitter<MemviewTableEditEvent> for MemviewTable {}

impl PchanAppActions for MemviewTable {
    fn appcx(&self) -> &AppCx {
        &self.appcx
    }
}
impl Render for MemviewTable {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let columns = 4u64;
        let items = u64::from(u32::MAX) / (columns * 4);
        let view = cx.entity();
        let theme = cx.theme().clone();
        let memviewer = VirtualList::new("memview-table", items, move |row_idx, _, cx| {
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
                    .appcx
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
                            .priority(1),
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
                    .appcx
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
                    (Some(edit), _) if edit.address == address => &theme.primary,
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
        .track_scroll(&self.scroll_handle);

        let view = cx.entity();
        let input = window.use_state(cx, |win, cx| {
            cx.subscribe_in(
                &view,
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
                let memview = view.clone();
                move |value, _, cx| -> Option<()> {
                    let value = value?;
                    memview.update(cx, |memview, cx| {
                        memview.scroll_handle.scroll_to(
                            cx,
                            u64::from(value) / 16,
                            ScrollStrategy::Top,
                        );
                    });
                    None
                }
            })
        });
        let theme = cx.theme();
        let jumpbar = Input::new(&input.read(cx).input)
            .font_family(&theme.mono_font_family)
            .prefix("Jump to: ");

        v_flex()
            .size_full()
            .child(jumpbar)
            .child(memviewer.flex_grow_1())
            .pchan_actions(self, cx)
            .wrap_window(cx)
            .title(Some("Memory"))
            .content(self.appcx.emucx.read(cx).content_title())
            .into_element()
            .text_color(cx.theme().foreground)
    }
}

struct SettingsView {
    appcx: AppCx,
    tab:   SettingsTab,
}

impl PchanAppActions for SettingsView {
    fn appcx(&self) -> &AppCx {
        &self.appcx
    }
}

impl SettingsView {
    fn is_on_tab(&self, tab: SettingsTab) -> bool {
        self.tab == tab
    }

    fn tab_button(&self, cx: &mut Context<Self>, tab: SettingsTab) -> Button {
        Button::new(ElementId::Name(format!("settings-tab-{tab:?}").into()))
            .ghost()
            .selected(self.is_on_tab(tab))
            .h_flex()
            .justify_start()
            .items_start()
            .child(
                div()
                    .size_full()
                    .h_flex()
                    .items_center()
                    .child(format!("{tab:?}")),
            )
            .when(self.is_on_tab(tab), |this| {
                this.border_color(cx.theme().foreground).border_2()
            })
            .on_click(cx.listener(move |settings, _, _, cx| {
                settings.tab = tab;
                cx.notify();
            }))
    }

    fn audio_tab(&self, win: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let slider = win.use_state(cx, |win, cx| {
            SliderState::new().min(0.).max(100.).default_value(100.)
        });
        let emucx = self.appcx.emucx.clone();
        win.use_state(cx, {
            let slider = &slider;
            move |_, cx| {
                cx.subscribe(slider, move |_, slider, event, cx| {
                    let SliderEvent::Change(value) = event else {
                        return;
                    };
                    let value = value.end() / 100.0;
                    emucx.update(cx, |emucx, _| {
                        emucx.emu.spu.app_volume = value;
                    });
                })
                .detach();
            }
        });
        div()
            .size_full()
            .v_flex()
            .gap_2()
            .p_2()
            .px_4()
            .child(div().child("Audio Settings").text_xl().mb_2())
            .child(
                h_flex()
                    .gap_2()
                    .child("Volume")
                    .child(
                        div()
                            .w_12()
                            .whitespace_nowrap()
                            .overflow_hidden()
                            .child(format!("{:.0}%", slider.read(cx).value())),
                    )
                    .child(Slider::new(&slider)),
            )
    }
}

impl Render for SettingsView {
    fn render(&mut self, win: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .border_t_1()
            .border_color(cx.theme().border)
            .size_full()
            .child(
                v_flex()
                    .flex_grow_1()
                    .p_2()
                    .h_full()
                    .max_w(rems(10.))
                    .border_r_1()
                    .border_color(cx.theme().border)
                    .gap_2()
                    .child(self.tab_button(cx, SettingsTab::General))
                    .child(self.tab_button(cx, SettingsTab::Audio)),
            )
            .child(div().flex_grow_1().h_full().child(match self.tab {
                SettingsTab::General => div().into_any_element(),
                SettingsTab::Audio => self.audio_tab(win, cx).into_any_element(),
            }))
            .pchan_actions(self, cx)
            .wrap_window(cx)
            .title(Some("Settings"))
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
            background_color: Some(cx.theme().primary.opacity(0.5)),
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
