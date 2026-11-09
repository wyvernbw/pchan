#![feature(duration_millis_float)]
#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::*;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::window::RequestRedraw;

mod emulator;
mod ui;

use crate::emulator::{DisplayNode, DisplayTarget, Psx, PsxState, emulator_plugin};
use crate::ui::{Theme, ui_plugin};

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
}

fn status_bar(theme: &Theme) -> impl Scene {
    let border = theme.border;
    bsn! {
        Node {
            height: rem(1.75),
            width: percent(100),
            border: UiRect::top(px(1)),
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            padding: UiRect::horizontal(rem(1)),
            align_items: AlignItems::Center
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
                width: rem(4.0)
            }
            @text("N/A ms")
            StatusbarItem::Elapsed
        ]
    }
}

fn app_ui_plugin(app: &mut App) {
    app.add_systems(Update, update_statusbar);
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
                let elapsed = psx.elapsed.iter().sum::<f32>() / (psx.elapsed.len() as f32);
                _ = write!(text.0, "{elapsed:.2}ms");
            }
        }
    }
}

fn text(value: impl AsRef<str>) -> impl Scene {
    let value = value.as_ref();
    bsn! {
        Text(value)
        TextFont {
            font: bevy::text::FontSourceTemplate::Generic(GenericFontFamily::SansSerif),
            font_size: px(14.),
        }
    }
}
