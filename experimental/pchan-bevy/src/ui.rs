use bevy::app::Propagate;
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::theme::{ThemeProps, UiTheme};
use bevy::feathers::{FeathersPlugins, tokens};
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use bevy::winit::WinitSettings;

pub fn ui_plugin(app: &mut App) {
    let theme = startup_get_theme();

    app.add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(dark_theme()))
        .insert_resource(WinitSettings::desktop_app())
        .insert_resource(theme)
        .add_observer(override_feathers_font_sans_serif)
        .add_observer(add_default_font_color)
        .add_observer(on_txt_color_added)
        .insert_resource(RemSize(16.0));
}

#[must_use]
pub fn startup_get_theme() -> Theme {
    let fg = Color::WHITE;

    Theme {
        fg,
        border: fg.darker(0.95),
    }
}

fn override_feathers_font_sans_serif(
    _: On<Add<Text>>,
    mut query: Query<&mut Propagate<TextFont>, Added<Propagate<TextFont>>>,
    mut ev: MessageWriter<RequestRedraw>,
) {
    for mut p in &mut query {
        p.0.font = FontSource::ui_sans_serif();
    }
    ev.write(RequestRedraw);
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
