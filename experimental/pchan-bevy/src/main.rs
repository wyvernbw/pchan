use bevy::app::Propagate;
use bevy::feathers::controls::{ButtonVariant, FeathersButton};
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::display::caption;
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::theme::{ThemeBackgroundColor, UiTheme};
use bevy::feathers::{FeathersPlugins, *};
use bevy::log::tracing;
use bevy::prelude::*;
use bevy::ui_widgets::Activate;
use bevy::window::RequestRedraw;
use bevy::winit::WinitSettings;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin::default()))
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(WinitSettings::desktop_app())
        .add_systems(Startup, setup.after(scene.spawn()))
        .add_systems(Startup, scene.spawn())
        .add_observer(
            |_: On<Add<Text>>,
             mut query: Query<&mut Propagate<TextFont>, Added<Propagate<TextFont>>>,
             mut ev: MessageWriter<RequestRedraw>| {
                for mut p in &mut query {
                    p.0.font = FontSource::ui_sans_serif();
                }
                ev.write(RequestRedraw);
            },
        )
        .run();
}

fn setup(mut commands: Commands, mut ev: MessageWriter<RequestRedraw>) {
    ev.write(RequestRedraw);
}

fn scene() -> impl SceneList {
    bsn_list! {
        Camera2d
        --
        @root()
    }
}

fn root() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            column_gap: px(8),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            @FeathersButton {
                @caption: bsn! { @caption("Normal") }
            }
            AccessibleLabel("Normal")
            on(|_: On<Activate>| {
                info!("Normal button clicked!");
            })
            --
            @FeathersButton {
                @caption: bsn! { @caption("Primary") },
                @variant: ButtonVariant::Primary,
            }
            AccessibleLabel("Primary")
            on(|_: On<Activate>| {
                info!("Primary button clicked!");
            })
        ]
    }
}
