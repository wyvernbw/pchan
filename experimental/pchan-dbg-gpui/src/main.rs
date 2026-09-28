#![allow(recursion_depth_exceeding_limit)]

#[path = "game-surface.rs"]
pub mod game_surface;

use core::num::ParseIntError;
use core::time::Duration;
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gpui::prelude::*;
use gpui::{AppContext, Render, *};
use gpui_base::{
    ElementExt, TextSelectionHandle, TextSelectionLayer, TextSelectionRegistration,
    TextSelectionRun,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::collapsible::Collapsible;
use gpui_component::input::{Input, InputState};
use gpui_component::menu::DropdownMenu;
use gpui_component::separator::Separator;
use gpui_component::spinner::Spinner;
use gpui_component::tab::TabBar;
use gpui_component::text::{TextView, markdown};
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

actions!(app, [Quit, SoftReset, HardReset]);

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> miette::Result<()> {
    init_tracing(pchan_utils::InitTracingArgs {
        stdout:     false,
        file:       true,
        panic_hook: false,
    });
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    #[cfg(feature = "dhat-heap")]
    let _profiler_ptr = &_profiler as *const dhat::Profiler as *mut dhat::Profiler;

    gpui_platform::application()
        .with_assets(PchanAssets::new())
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(move |cx| {
            gpui_component::init(cx);

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
                    let view = Debugger::new(win, cx).unwrap();
                    let view = cx.new(|_| view);
                    {
                        // let view = view.clone();
                        // win.on_next_frame(move |_, cx| {
                        //     view.update(cx, |view, _| {
                        //         let mem_idx = 0xbfc0_0000 / 16;
                        //         view.memview_scroll
                        //             .scroll_to_item(mem_idx, ScrollStrategy::Top);
                        //     });
                        // });
                    }
                    let theme = cx.theme().clone();

                    cx.new(|cx| Root::new(view, win, cx).h_full().bg(theme.background))
                },
            );

            cx.activate(true);
        });

    Ok(())
}

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
    game_surface: Entity<GameSurface>,

    cached_reg_names:    [SharedString; 32],
    cpu_control_reg_tab: usize,

    exec_control_panel_open: bool,
    mips_dump_scroll_handle: UniformListScrollHandle,
    disc_path:               Option<PathBuf>,

    memview: Entity<MemviewTable>,

    target:     PchanTexture,
    target_buf: wgpu::Buffer,
}

struct EmuContext {
    emu:            Emu,
    runner:         Runner,
    running:        bool,
    running_notify: event_listener::Event,
    renderer:       Arc<pchan_gpu::Renderer>,
    frame_time:     Duration,
}

use miette::{IntoDiagnostic, miette};
use pchan_emu::run::Runner;
use pchan_gpu::wgpu;

use crate::game_surface::{GameSurface, PchanTexture, SurfaceState, create_target};

impl Debugger {
    pub fn new(_window: &Window, cx: &mut App) -> miette::Result<Self> {
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
        drop(dp);

        gpu.connect_emu(&mut emu);
        let gpu = Arc::new(gpu);
        gpu.clone().start();

        let (target, target_buf) = create_target(&gpu, &mut gpu.display_uniforms.lock().unwrap());

        let cached_reg_names = core::array::from_fn(|reg| {
            let reg = match reg as u8 {
                0 => "0",
                pchan_emu::cpu::FP => "fp",
                other => REG_STR[other as usize],
            };
            format!("${}", reg).into()
        });

        let emucx = cx.new(|_| EmuContext {
            emu,
            renderer: gpu.clone(),
            running: false,
            running_notify: event_listener::Event::new(),
            runner: Runner::new().with_config(pchan_emu::run::RunnerConfig {
                force_mode: Some(pchan_emu::run::RunnerMode::Dynarec),
            }),
            frame_time: Duration::ZERO,
        });

        let surface_state = cx.new(|_| SurfaceState::new(target.clone(), target_buf.clone()));
        let game_surface = cx.new(|_| {
            GameSurface::new(
                "global-game-surface",
                gpu.clone(),
                true,
                surface_state.clone(),
            )
        });

        let memview = cx.new(|_| MemviewTable {
            scroll: UniformListScrollHandle::new(),
            emucx:  emucx.clone(),
        });

        cx.spawn_with_priority(Priority::High, {
            let surface = game_surface.clone();
            let surface_state = surface_state.clone();
            let emucx = emucx.clone();
            async move |cx| {
                loop {
                    let run_listener = cx.read_entity(&emucx, |emucx, _| match emucx.running {
                        false => Some(emucx.running_notify.listen()),
                        true => None,
                    });
                    if let Some(run_listener) = run_listener {
                        run_listener.await;
                    }
                    let start = Instant::now();
                    let frame_time = emucx.update(cx, |emucx, cx| {
                        if emucx.running {
                            let surface_state = surface_state.as_mut(cx);
                            surface_state.start_display_draw(&emucx.renderer);

                            while !emucx.emu.consume_vblank_signal() {
                                emucx.runner.execute(&mut emucx.emu);
                            }

                            surface_state.wait_for_display_draw(&emucx.renderer);
                            surface_state.start_convert_render(&emucx.renderer);
                            drop(surface_state);
                            surface.update(cx, |_, cx| cx.notify());
                        }
                        let frame_time = start.elapsed();
                        emucx.frame_time = frame_time;

                        frame_time
                    });

                    spin_sleep(cx, Duration::from_micros(16_667).saturating_sub(frame_time)).await;
                }
            }
        })
        .detach();

        cx.on_action::<HardReset>({
            let emucx = emucx.clone();
            let surface = game_surface.clone();
            let surface_state = surface_state.clone();
            move |_, cx| {
                emucx
                    .update(cx, |emucx, cx| -> miette::Result<()> {
                        let bios_path = emucx.emu.bootloader().bios_path.clone();
                        emucx.emu = Emu::new();
                        emucx.emu.set_bios_path(bios_path);
                        emucx.emu.load_bios().into_diagnostic()?;
                        emucx.emu.gpu.vram = pchan_emu::gpu::create_vram();
                        emucx.emu.cpu.jump_to_bios();
                        emucx.renderer.reset();
                        emucx.renderer.connect_emu(&mut emucx.emu);
                        emucx.emu.tty.set_tracing();

                        let mut audio_task = AudioTask::new()?;
                        pchan_bind::bind_audio(&mut audio_task, &mut emucx.emu);
                        let audio_stream = audio_task.start()?;
                        std::mem::forget(audio_stream);

                        Ok(())
                    })
                    .unwrap();
            }
        });

        Ok(Self {
            emucx,
            game_surface,

            memview,

            cached_reg_names,
            cpu_control_reg_tab: 0,
            exec_control_panel_open: true,
            disc_path: None,
            mips_dump_scroll_handle: UniformListScrollHandle::new(),

            target,
            target_buf,
        })
    }
}

async fn spin_sleep(cx: &AsyncApp, duration: Duration) {
    let sleep_for = duration.saturating_sub(Duration::from_millis(4));
    let deadline = Instant::now() + duration;

    cx.background_executor()
        .spawn_with_priority(Priority::High, cx.background_executor().timer(sleep_for))
        .await;
    loop {
        if Instant::now() > deadline {
            return;
        }
        std::hint::spin_loop();
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

        v_flex()
            .child(TextSelectionLayer)
            .h(window_height)
            .child(header(cx, self.emucx.read(cx).frame_time))
            .child(
                div()
                    .h_full()
                    .bg(transparent_white())
                    .child(
                        div()
                            .absolute()
                            .w_full()
                            .h_full()
                            .child(self.game_surface().clone()),
                    )
                    .child(self.debugger_ui(window, cx).w_full().h_full()),
            )
    }
}

fn header<T: 'static>(cx: &Context<T>, frame_time: Duration) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .text_sm()
        .bg(theme.title_bar)
        .px_4()
        .pl(rems(4.5))
        .py_1()
        .gap_4()
        .items_center()
        .border_color(theme.title_bar_border)
        .border_b_1()
        .text_color(theme.table_head_foreground)
        .font_family(&theme.mono_font_family)
        .child("🐷🎗️ P-ちゃん")
        .child(div().flex_grow_1())
        .child(markdown(format!("frame: {:02}ms", frame_time.as_millis())))
}

impl Debugger {
    fn debugger_ui(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + Styled {
        let theme = cx.theme().clone();

        v_flex()
            .id("main")
            .w_full()
            .h_full()
            .relative()
            .text_color(theme.foreground)
            .p_4()
            .gap_4()
            .bg(transparent_white())
            .child(
                div().h_flex().items_start().flex_grow_1().text_sm().child(
                    panel(&theme)
                        .v_flex()
                        .flex_grow_1()
                        .max_w_96()
                        .when(self.exec_control_panel_open, |this| this.min_h_full())
                        .gap_2()
                        .child(
                            h_flex()
                                .justify_between()
                                .gap_2()
                                .child(self.execution_header(cx, &theme).min_w_0().flex_grow_1())
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
                            .gap_2()
                            .child(
                                div()
                                    .h_flex()
                                    .items_center()
                                    .flex_grow_0()
                                    .gap_2()
                                    .child(markdown("Registers").text_color(theme.muted_foreground))
                                    .child(self.cpu_control_tabbar(cx)),
                            )
                            .child(self.cpu_controls(window, cx).min_h_0().flex_grow_1()),
                    )
                    .child(
                        v_flex()
                            .flex_grow_1()
                            .h_full()
                            .min_h_0()
                            .gap_2()
                            .child(
                                h_flex().h_8().child(
                                    div().child("Memory").text_color(theme.muted_foreground),
                                ),
                            )
                            .child(
                                panel(&theme)
                                    .flex_grow_1()
                                    .min_h_0()
                                    .child(self.memview.clone()),
                            ),
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
        .p_2()
}

fn parse_hex_register(str: &str) -> Result<u32, ParseIntError> {
    if str == "0x" {
        return Ok(0);
    }
    u32::from_str_radix(str.trim_prefix("0x"), 16)
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
                            let reg_value = match parse_hex_register(&input_state.read(cx).value())
                            {
                                Ok(reg_value) => {
                                    view.emucx
                                        .update(cx, |emucx, _| emucx.emu.cpu.gpr[r] = reg_value);
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
                )
                .detach();

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
                    .h(rems(1.))
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

        panel(&theme)
            .v_flex()
            .id("cpu-scroll-container")
            .gap_2()
            .child(
                div()
                    .gap_1()
                    .v_flex()
                    .flex_wrap()
                    .min_h_0()
                    .flex_grow_1()
                    .w_full()
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
        let ix = (pc & 0x1fffffff) / 4;

        if self.emucx.read(cx).running {
            self.mips_dump_scroll_handle
                .scroll_to_item(ix as usize, ScrollStrategy::Center);
        }

        v_flex()
            .overflow_y_hidden()
            .gap_2()
            .child(
                h_flex()
                    .justify_between()
                    .child(
                        markdown("MIPS dump")
                            .text_color(theme.muted_foreground)
                            .flex_grow_0(),
                    )
                    .child(
                        markdown(format!("$pc: {}", hex(pc)))
                            .selectable(true)
                            .font_family(&theme.mono_font_family),
                    ),
            )
            .child(
                uniform_list("mips-dump", u32::MAX as usize / 4, move |range, _, cx| {
                    let theme = &theme;
                    cx.read_entity(&entity, move |view, cx| {
                        range
                            .map(|idx| {
                                let address_label: SharedString = "mips-dump-address".into();
                                let op_label: SharedString = "mips-dump-op".into();

                                let addr = idx as u32 * 4;
                                let instr = view.emucx.read(cx).emu.fastmem_read::<OpCode>(addr);
                                let instr = instr
                                    .map(DecodedOp::new)
                                    .map(|instr| Cow::Owned(format!("{instr}")))
                                    .unwrap_or(Cow::Borrowed("N/A"));
                                let is_pc = pc & 0x1fff_ffff == addr & 0x1fff_ffff;
                                h_flex()
                                    .w_full()
                                    .font_family(&theme.mono_font_family)
                                    .bg(theme.foreground.opacity(if idx.is_multiple_of(2) {
                                        0.0
                                    } else {
                                        0.08
                                    }))
                                    .child(
                                        TextView::markdown(
                                            ElementId::NamedInteger(op_label, idx as u64),
                                            hex(addr).to_string(),
                                        )
                                        .selectable(true)
                                        .opacity(0.5),
                                    )
                                    .child(
                                        div()
                                            .text_center()
                                            .w_4()
                                            .when(is_pc, |this| this.child(">")),
                                    )
                                    .child(
                                        TextView::markdown(
                                            ElementId::NamedInteger(address_label, idx as u64),
                                            instr,
                                        )
                                        .selectable(true),
                                    )
                                    .when(is_pc, |this| this.text_color(theme.colors.info))
                                    .h_4()
                            })
                            .collect()
                    })
                })
                .track_scroll(&self.mips_dump_scroll_handle)
                .corner_radii(Corners::all(8.0.into()))
                .flex_grow_1(),
            )
    }

    fn open_disc_button(&mut self, cx: &mut Context<Debugger>, _theme: &Theme) -> Button {
        Button::new("disc-path-button")
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
                    let disc_path = res
                        .pop()
                        .ok_or_else(|| miette!("expected at least one disc path"))?;

                    view.update(cx, move |view, cx| -> miette::Result<()> {
                        view.emucx.update(cx, |emucx, _| -> miette::Result<()> {
                            let fsm = emucx.emu.open_disc(&disc_path, false).into_diagnostic()?;
                            emucx
                                .emu
                                .advance_open_disc(&disc_path, fsm, false)
                                .into_diagnostic()?;
                            Ok(())
                        })?;
                        view.disc_path = Some(disc_path);
                        Ok(())
                    })?;

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
        let open_disc = self.open_disc_button(cx, theme);
        match self.disc_path.as_ref() {
            None => h_flex().flex_grow_1().child(
                open_disc
                    .label("Load Disc")
                    .icon(Icon::empty().path("disc-3.svg")),
            ),
            Some(path) => h_flex()
                .min_w_0()
                .gap_2()
                .w_full()
                .child(
                    div().min_w_0().flex_grow_1().child(
                        open_disc.w_full().text_ellipsis().flex().label(
                            path.file_name()
                                .map(|f| f.to_string_lossy())
                                .unwrap_or(Cow::Borrowed("Unknown")),
                        ),
                    ),
                )
                .child(Button::new("eject-disc-button").icon(Icon::empty().path("eject.svg"))),
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
            .min_w_0()
            .flex_grow_1()
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
        div()
            .v_flex()
            .gap_2()
            .child(Separator::horizontal())
            .child(self.instructions_list(cx, theme).flex_grow_1().min_h_0())
    }
}

struct MemviewTable {
    scroll: UniformListScrollHandle,
    emucx:  Entity<EmuContext>,
}

impl Render for MemviewTable {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let columns = 4;
        let items = u32::MAX as usize / (columns * 4);
        let view = cx.entity();
        let theme = cx.theme().clone();
        uniform_list("memview-table", items, move |range, win, cx| {
            range
                .map(|row_idx| {
                    use core::fmt::Write;

                    let caddress = hex(row_idx as u32 * columns as u32 * 4);

                    let mut result = h_flex().font_family(&theme.mono_font_family).gap_2().child(
                        TextView::markdown(
                            ElementId::NamedInteger("mem-view-row-address".into(), row_idx as u64),
                            caddress.as_str(),
                        ),
                    );

                    for word_idx in 0..columns {
                        let address = row_idx * columns * 4 + word_idx * 4;
                        let address = address & 0x1fff_ffff;
                        let word = view
                            .read(cx)
                            .emucx
                            .read(cx)
                            .emu
                            .try_read_pure::<u32>(address as u32)
                            .unwrap_or(0);
                        let hex = hex_pref::<_, false>(word);

                        let id =
                            ElementId::NamedInteger("mewmview-hex-input".into(), address as u64);
                        let input_state = win.use_keyed_state(id, cx, |win, cx| {
                            InputState::new(win, cx).default_value(hex.as_str())
                        });
                        win.subscribe(&input_state, cx, {
                            let emucx = view.read(cx).emucx.clone();
                            let input_state = input_state.clone();
                            use gpui_component::input::InputEvent;
                            move |_, event: &InputEvent, win, cx| {
                                if let InputEvent::PressEnter { .. } | InputEvent::Blur = event {
                                    let mem_value =
                                        match parse_hex_register(&input_state.read(cx).value()) {
                                            Ok(mem_value) => {
                                                emucx.update(cx, |emucx, _| {
                                                    emucx.emu.write(address as u32, mem_value);
                                                });
                                                mem_value
                                            }
                                            // TODO: handle error
                                            Err(_err) => return,
                                        };
                                    input_state.update(cx, |state, cx| {
                                        state.set_value(
                                            hex_pref::<_, false>(mem_value).as_str(),
                                            win,
                                            cx,
                                        );
                                    });
                                }
                            }
                        })
                        .detach();

                        if !input_state.focus_handle(cx).is_focused(win) {
                            input_state.update(cx, |state, cx| {
                                state.set_value(hex.as_str(), win, cx);
                            });
                        }

                        result = result.child(Input::new(&input_state).text_center().w(rems(6.)));
                    }

                    for word_idx in 0..columns {
                        let address = row_idx * columns * 4 + word_idx * 4;
                        let address = address & 0x1fff_ffff;
                        let mut word = view
                            .read(cx)
                            .emucx
                            .read(cx)
                            .emu
                            .try_read_pure::<[u8; 4]>(address as u32)
                            .unwrap_or([b'.'; 4]);
                        for byte in word.iter_mut() {
                            match *byte {
                                ..=0x1f | 0x7f.. => {
                                    *byte = b'.';
                                }
                                _ => {}
                            }
                        }
                        let word = core::str::from_utf8(&word).expect("impossible");

                        let id =
                            ElementId::NamedInteger("mewmview-hex-ascii".into(), address as u64);
                        result = result.child(TextView::markdown(id, word));
                    }

                    result
                })
                .collect()
        })
        .w_full()
        .h_full()
        .track_scroll(&self.scroll)
    }
}
