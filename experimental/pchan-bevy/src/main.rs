#![feature(duration_millis_float)]
#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

use core::time::Duration;

use bevy::ecs::system::NonSendMarker;
use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::*;
use bevy::log::{LogPlugin, tracing};
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, RequestRedraw, WindowEvent};
#[cfg(target_os = "macos")]
use bevy::winit::WINIT_WINDOWS;
use bevy::winit::{RawWinitWindowEvent, WinitWindows};

mod emulator;
mod ui;

use crate::emulator::{DisplayNode, DisplayTarget, Psx, PsxState, emulator_plugin};
use crate::ui::{Theme, text, ui_plugin, vseparator};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    pchan_utils::init_tracing(pchan_utils::InitTracingArgs {
        stdout:     true,
        file:       true,
        panic_hook: false,
    });
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    App::new()
        .add_plugins(
            DefaultPlugins
                .build()
                .disable::<LogPlugin>()
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        titlebar_shown: true,
                        titlebar_transparent: true,
                        fullsize_content_view: true,
                        title: "🐷🎗️P-ちゃん".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(OnExit(PsxState::Setup), setup)
        .add_plugins(ui_plugin)
        .add_plugins(emulator_plugin)
        .add_plugins(app_ui_plugin)
        .run();
}

fn setup(
    mut cmds: Commands,
    mut ev: MessageWriter<RequestRedraw>,
    theme: Res<Theme>,
    display: Res<DisplayTarget>,
) {
    cmds.spawn_scene_list(bsn_list! {
         Camera2d
         --
         @root(&theme, &display.0)
    });
    ev.write(RequestRedraw);
}

fn on_window_event(
    mut evs: MessageReader<WindowEvent>,
    mut statusbar: Query<&mut Node, With<Statusbar>>,
    _non_send_marker: NonSendMarker,
) {
    for ev in evs.read() {
        if let WindowEvent::WindowResized(ev) = ev {
            WINIT_WINDOWS.with_borrow_mut(|windows| {
                if let Some(win) = windows.get_window(ev.window) {
                    let display = match (win.is_maximized(), win.fullscreen()) {
                        // maximizing on macos is actually full screen
                        #[cfg(target_os = "macos")]
                        (true, _) => Display::None,
                        (_, Some(_)) => Display::None,
                        _ => Display::Flex,
                    };

                    for mut s in &mut statusbar {
                        s.display = display;
                    }
                }
            });
        }
    }
}

fn root(theme: &Theme, img: &Handle<Image>) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            column_gap: px(8),
            display: Display::Flex,
            flex_direction: FlexDirection::Column
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            Node {
                height: rem(1.75),
                width: percent(100),
                padding: UiRect::horizontal(rem(1))
            }
            --
            Node {
                width: percent(100.),
                flex_grow: 1.0
            }
            DisplayNode
            Children [
                Node {
                    position_type: PositionType::Absolute,
                    width: percent(100),
                    height: percent(100)
                }
                ImageNode {
                    image: {img.clone()}
                }
            ]
            --
            @status_bar(theme)
        ]
    }
}

#[derive(Clone, Copy, Component, Default)]
enum StatusbarItem {
    #[default]
    None,
    FrameCounter,
    Elapsed,
    DisplayInfo,
    SpeedAndFps,
}

#[derive(Component, Clone, Copy, Default)]
pub struct Statusbar;

fn status_bar(theme: &Theme) -> impl Scene {
    let border = theme.border;
    bsn! {
        Statusbar
        Node {
            height: rem(1.75),
            width: percent(100),
            border: UiRect::top(px(1)),
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            padding: UiRect::horizontal(rem(1)),
            align_items: AlignItems::Center,
            column_gap: rem(1)
        }
        BorderColor::all(border)
        Children [
            Node {
                min_width: rem(8.0),
                display: Display::Flex
            }
            StatusbarItem::FrameCounter
            @text("#frame")
            --
            Node {
                flex_grow: 1.0
            }
            --
            Node {
                width: rem(7.0)
            }
            @text("N/A")
            StatusbarItem::DisplayInfo
            --
            @vseparator(theme)
            --
            Node {
                width: rem(7.0)
            }
            @text("Paused")
            StatusbarItem::SpeedAndFps
            --
            @vseparator(theme)
            --
            Node {
                width: rem(4.0)
            }
            @text("N/A ms")
            StatusbarItem::Elapsed
            --
        ]
    }
}

fn app_ui_plugin(app: &mut App) {
    app.add_systems(Update, (update_statusbar,))
        .add_systems(PostUpdate, (on_window_event,))
        .add_systems(PreUpdate, (keyboard_shortcuts,));
}

fn update_statusbar(psx: If<Res<Psx>>, items: Query<(&mut Text, &StatusbarItem)>) {
    use core::fmt::Write;

    for (mut text, item) in items {
        text.0.clear();
        match item {
            StatusbarItem::None => {}
            StatusbarItem::FrameCounter => {
                _ = write!(text.0, "#{}", psx.runner.frame_idx);
            }
            StatusbarItem::Elapsed => {
                if psx.elapsed.is_empty() {
                    _ = write!(text.0, "Stopped");
                } else {
                    let elapsed = psx.elapsed.iter().sum::<f32>() / (psx.elapsed.len() as f32);
                    _ = write!(text.0, "{elapsed:.2}ms");
                }
            }
            StatusbarItem::DisplayInfo => {
                let dp_mode = psx.emu.gpu.gpustat.video_mode();
                let res = psx.emu.gpu.gpustat.resolution();
                _ = write!(text.0, "{}x{} {dp_mode}", res.x, res.y)
            }
            StatusbarItem::SpeedAndFps => {
                let elapsed = *psx.elapsed.back().unwrap_or(&0.0);
                let refresh_rate = psx.emu.gpu.gpustat.refresh_rate();
                let target_fps = psx.emu.gpu.gpustat.frame_target();
                let time_per_frame = 1000.0 / refresh_rate;
                let speed = time_per_frame / elapsed;
                let speed_percentage = (speed * 100.).clamp(0.0, 100.);
                let fps = speed_percentage * f32::from(target_fps) / 100.;
                _ = write!(text.0, "{fps:.0}FPS ({speed_percentage:.0}%)");
            }
        }
    }
}

fn keyboard_shortcuts(
    mut cmds: Commands,
    mut psx: If<ResMut<Psx>>,
    psx_state: Res<State<PsxState>>,
    input: Res<ButtonInput<KeyCode>>,
    mut ev: MessageWriter<RequestRedraw>,
) {
    let psx_state = *psx_state.get();
    let cmd = if cfg!(target_os = "macos") {
        input.pressed(KeyCode::SuperLeft)
    } else {
        input.pressed(KeyCode::ControlLeft)
    };

    if cmd && input.just_pressed(KeyCode::KeyJ) {
        psx.toggle_running(&mut cmds);
    }

    match psx_state {
        PsxState::Paused => {
            if cmd && input.pressed(KeyCode::KeyK) {
                psx.run_one_frame(&mut cmds);
                ev.write(RequestRedraw);
            }
        }
        _ => {}
    }
}
