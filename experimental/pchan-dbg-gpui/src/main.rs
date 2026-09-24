use std::rc::Rc;

use gpui::prelude::*;
use gpui::*;
use gpui::{AppContext, Render};
use gpui_component::text::markdown;
use gpui_component::{ActiveTheme, Root, Theme, ThemeConfig, button::*};

pub type Resul<T> = Result<T, Box<dyn std::error::Error>>;

actions!(app, [Quit]);

fn main() {
    gpui_platform::application()
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(|cx| {
            gpui_component::init(cx);
            let theme = Theme::global_mut(cx);
            theme.apply_config(&Rc::new(ThemeConfig {
                mode: gpui_component::ThemeMode::Dark,
                ..Default::default()
            }));
            theme.background = theme.background.opacity(0.8);

            cx.set_window_appearance(Some(WindowAppearance::VibrantDark));

            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.on_action(|_: &Quit, cx| {
                cx.quit();
            });

            cx.spawn(async move |cx| {
                _ = cx.open_window(
                    gpui::WindowOptions {
                        window_background: WindowBackgroundAppearance::Blurred,
                        ..Default::default()
                    },
                    |win, cx| {
                        let view = cx.new(|_| HelloWorld);
                        let theme = cx.theme().clone();
                        cx.new(|cx| Root::new(view, win, cx).bg(theme.background))
                    },
                );
            })
            .detach();
            cx.activate(true);
        });
}

struct HelloWorld;

impl Render for HelloWorld {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::prelude::Context<Self>,
    ) -> impl gpui::prelude::IntoElement {
        let theme = cx.theme();
        div()
            .id("main")
            .text_color(theme.foreground)
            .p_4()
            .flex_col()
            .overflow_scroll()
            .child(markdown("Hello world!").selectable(true))
            .child(Button::new("Wow!").child("Wow!"))
    }
}
