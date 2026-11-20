use core::time::Duration;

use bevy::app::Propagate;
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::theme::{ThemeProps, UiTheme};
use bevy::feathers::{FeathersPlugins, tokens};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::window::RequestRedraw;
use bevy::winit::{UpdateMode, WinitSettings};

pub fn ui_plugin(app: &mut App) {
    let theme = startup_get_theme();

    let on_input = UpdateMode::Reactive {
        wait:                   Duration::MAX,
        react_to_device_events: false,
        react_to_user_events:   true,
        react_to_window_events: true,
    };

    app.add_plugins(FeathersPlugins)
        .add_message::<RerenderMsg>()
        .insert_resource(UiTheme(dark_theme()))
        .insert_resource(WinitSettings {
            focused_mode:   on_input,
            unfocused_mode: on_input,
        })
        .insert_resource(theme)
        .add_observer(override_feathers_font_sans_serif)
        .add_observer(add_default_font_color)
        .add_observer(on_txt_color_added)
        .add_systems(Update, rerender_check)
        // this should run after rerender_check on the next frame
        .add_systems(PreUpdate, reset_cameras)
        .insert_resource(RemSize(16.0));

    if let Some(subapp) = app.get_sub_app_mut(RenderApp) {
        subapp.add_message::<RerenderMsg>();
    }
}

#[derive(Component, Clone, Copy, Default)]
pub struct UiCamera;

#[derive(Message, Clone, Copy, Default)]
pub struct RerenderMsg;

pub fn rerender_check(
    mut query_cams: Query<&mut Camera, With<UiCamera>>,
    mut reader: MessageReader<RerenderMsg>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    if !reader.is_empty() {
        for mut cam in query_cams.iter_mut() {
            cam.is_active = true;
        }
        reader.clear();
        redraw.write(RequestRedraw);
    }
}

pub fn reset_cameras(query_cams: Query<&mut Camera, With<UiCamera>>) {
    for mut cam in query_cams {
        cam.is_active = false;
    }
}

#[must_use]
pub fn startup_get_theme() -> Theme {
    let fg = Color::WHITE;

    Theme {
        fg,
        border: fg.darker(0.95),
        bg: Color::BLACK,
    }
}

fn override_feathers_font_sans_serif(
    _: On<Add<Text>>,
    mut query: Query<&mut Propagate<TextFont>, Added<Propagate<TextFont>>>,
    mut ev: MessageWriter<RerenderMsg>,
) {
    for mut p in &mut query {
        p.0.font = FontSource::ui_sans_serif();
    }
    ev.write(RerenderMsg);
}

#[derive(Component)]
pub struct TxtColor(pub Color);

fn add_default_font_color(
    _: On<Add<Text>>,
    theme: Res<Theme>,
    query: Query<Entity, (Added<Text>, Without<TxtColor>)>,
    mut cmd: Commands,
) {
    for e in query {
        cmd.entity(e).insert(TxtColor(theme.fg));
    }
}

fn on_txt_color_added(
    _: On<Add<TxtColor>>,
    query: Query<(Entity, &TxtColor), Added<TxtColor>>,
    mut cmds: Commands,
) {
    for (eid, color) in query {
        cmds.entity(eid).insert(TextColor(color.0));
    }
}

#[must_use]
pub fn dark_theme() -> ThemeProps {
    let mut theme = create_dark_theme();
    if let Some(bg_token) = theme.token_assignments.get(&tokens::WINDOW_BG) {
        theme
            .semantic_base
            .entry(bg_token.clone())
            .and_modify(|color| *color = Color::BLACK);
    }
    theme
}

#[derive(Resource)]
pub struct Theme {
    pub fg:     Color,
    pub bg:     Color,
    pub border: Color,
}

pub fn text(value: impl AsRef<str>) -> impl Scene {
    let value = value.as_ref();
    bsn! {
        Text(value)
        TextFont {
            font: bevy::text::FontSourceTemplate::Generic(GenericFontFamily::SystemUi),
            font_size: px(14.),
            weight: FontWeight(0),
        }
    }
}

pub fn vseparator(theme: &Theme) -> impl Scene {
    bsn! {
        Node {
            height: percent(100.),
            width: px(1.),
        }
        BackgroundColor({theme.border})
    }
}
