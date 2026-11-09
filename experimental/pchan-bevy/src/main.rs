use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::*;
use bevy::prelude::*;
use bevy::window::RequestRedraw;

mod ui;

use crate::ui::{Theme, ui_plugin};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                titlebar_shown: true,
                titlebar_transparent: true,
                fullsize_content_view: true,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_plugins(ui_plugin)
        .run();
}

fn setup(mut cmds: Commands, mut ev: MessageWriter<RequestRedraw>, theme: Res<Theme>) {
    cmds.spawn_scene_list(bsn_list! {
         Camera2d
         --
         @root(&theme)
    });
    cmds.spawn_scene(root(&theme));
    ev.write(RequestRedraw);
}

fn root(theme: &Theme) -> impl Scene {
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
                flex_grow: 1.0
            }
            --
            @status_bar(theme)
        ]
    }
}

fn status_bar(theme: &Theme) -> impl Scene {
    let border = theme.border;
    bsn! {
        Node {
            height: rem(1.25),
            width: percent(100),
            border: UiRect::top(px(1)),
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            padding: UiRect::horizontal(rem(1))
        }
        BorderColor::all(border)
        Children [
            Node {}
            @text("#frame")
        ]
    }
}

fn text(value: impl AsRef<str>) -> impl Scene {
    let value = value.as_ref();
    bsn! {
        Text(value)
        TextFont {
            font: bevy::text::FontSourceTemplate::Family("sans-serif"),
            font_size: rem(1)
        }
    }
}
