#![feature(duration_millis_float)]
#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

use std::collections::HashMap;
use std::path::PathBuf;

use bevy::ecs::system::NonSendMarker;
use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::*;
use bevy::input::gamepad::GamepadConnectionEvent;
use bevy::log::{LogPlugin, tracing};
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use bevy::window::{PrimaryWindow, RequestRedraw, WindowEvent};
#[cfg(target_os = "macos")]
use bevy::winit::WINIT_WINDOWS;
use bevy::winit::{EventLoopProxyWrapper, RawWinitWindowEvent, WinitWindows};
use bevy::{window, winit};
use muda::accelerator::{Accelerator, Code, Modifiers};
use muda::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use pchan_bind::input::PchanButton;
use pchan_emu::io::sio::Sio0Port;
use pchan_utils::{AsyncChan, Chan, kanal};
use rfd::AsyncFileDialog;

mod emulator;
mod ui;

use crate::emulator::{DisplayNode, DisplayTarget, Psx, PsxCmdMsg, PsxState, emulator_plugin};
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

    let mut title = String::new();
    app_title(&mut title, None);
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
                        title,
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

pub fn app_title(sink: &mut String, playing: Option<&str>) {
    use core::fmt::Write;
    use pchan_macros::git_rev;

    sink.clear();
    match playing {
        Some(game) => {
            let _ = write!(sink, "🐷🎗️P-ちゃん({}) {game}", git_rev!());
        }
        None => {
            let _ = write!(sink, "🐷🎗️P-ちゃん({})", git_rev!());
        }
    }
}

#[derive(Resource, Clone)]
struct MenubarChan(Chan<muda::MenuEvent>);

#[derive(Message, Clone, Copy)]
enum MainMenubarMsg {
    OpenContent,
    UnloadContent,
    Quit,
}

#[derive(Deref)]
struct NativeMenubar<Ev> {
    #[deref]
    menu:      Menu,
    event_map: HashMap<muda::MenuId, Ev>,
}

impl<Ev> NativeMenubar<Ev> {
    fn new() -> Self {
        Self {
            menu:      Menu::new(),
            event_map: HashMap::new(),
        }
    }
    fn get_event(&self, ev: muda::MenuEvent) -> Option<&Ev> {
        self.event_map.get(ev.id())
    }
    fn menu_item(&mut self, item: MenuItem, ev: Ev) -> MenuItem {
        let id = item.id();
        self.event_map.insert(id.clone(), ev);
        item
    }

    fn init(&self, #[allow(unused)] win: &window::RawHandleWrapper) {
        #[cfg(target_os = "windows")]
        unsafe {
            let win_handle = win.get_window_handle();
            self.menu.init_for_hwnd(window_hwnd)
        };

        #[cfg(target_os = "macos")]
        self.menu.init_for_nsapp();
    }
}

fn build_main_menubar(psx: &Psx) -> NativeMenubar<MainMenubarMsg> {
    let cmd = cfg_select! {
        target_os = "macos" => Modifiers::META,
        _ => Modifiers::CONTROL,
    };

    let mut menu = NativeMenubar::<MainMenubarMsg>::new();
    let open_content = menu.menu_item(
        MenuItem::new(
            "Open Content",
            !psx.has_content(),
            Some(Accelerator::new(cmd, Code::KeyO)),
        ),
        MainMenubarMsg::OpenContent,
    );
    let unload_content = menu.menu_item(
        MenuItem::new("Unload Content", psx.has_content(), None),
        MainMenubarMsg::UnloadContent,
    );
    let quit = menu.menu_item(
        MenuItem::new("Quit", true, Some(Accelerator::new(cmd, Code::KeyQ))),
        MainMenubarMsg::Quit,
    );
    let submenu = Submenu::with_items(
        "",
        true,
        &[
            &open_content,
            &unload_content,
            &PredefinedMenuItem::separator(),
            &quit,
            // &menu_item2,
            // &MenuItem::new("Menu item #3", true, None),
            // &PredefinedMenuItem::separator(),
            // &Submenu::with_items(
            //     "Submenu Inner",
            //     true,
            //     &[
            //         &MenuItem::new("Submenu item #1", true, None),
            //         &PredefinedMenuItem::separator(),
            //         &menu_item2,
            //     ],
            // )
            // .unwrap(),
        ],
    )
    .unwrap();
    menu.append(&submenu).unwrap();

    menu
}

fn setup(world: &mut World) -> Result<()> {
    let display = world.resource::<DisplayTarget>();
    let theme = world.resource::<Theme>();
    world
        .spawn_scene_list(bsn_list! {
             Camera2d
             --
             @root(&theme, &display.0)
        })
        .unwrap();

    world.write_message(RequestRedraw);

    world.insert_resource(AsyncTaskChan(kanal::bounded_async(1)));

    tracing::info!("adding menus.");

    #[allow(unused)]
    let win = world
        .query_filtered::<&window::RawHandleWrapper, With<PrimaryWindow>>()
        .single(world)?;

    #[cfg(any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    todo!("in-engine menubar because linux sucks (jk ily)");

    let psx = world.resource::<Psx>();
    let menu = build_main_menubar(psx);
    menu.init(win);
    world.insert_non_send(menu);

    let menu_chan = kanal::bounded::<muda::MenuEvent>(4);
    world.insert_resource(MenubarChan(menu_chan.clone()));

    let event_loop_proxy = world.resource::<EventLoopProxyWrapper>();
    let proxy = winit::EventLoopProxy::clone(event_loop_proxy);
    muda::MenuEvent::set_event_handler(Some(move |event| {
        let _ = proxy.send_event(bevy::winit::WinitUserEvent::WakeUp);
        let _ = menu_chan.0.send(event);
    }));

    Ok(())
}

#[derive(Message, Clone, Copy, Default)]
pub struct UpdateMainMenubarMsg;

fn update_main_menubar(
    mut menubar: If<NonSendMut<NativeMenubar<MainMenubarMsg>>>,
    psx: If<Res<Psx>>,
    mut reader: MessageReader<UpdateMainMenubarMsg>,
    win: If<Single<&window::RawHandleWrapper, With<PrimaryWindow>>>,
) {
    for _ in reader.read() {
        **menubar = build_main_menubar(&psx);
        menubar.init(&win);
    }
}

fn menubar_channel_forward(
    chan: If<Res<MenubarChan>>,
    mut menu_tx: MessageWriter<MainMenubarMsg>,
    menu: If<NonSend<NativeMenubar<MainMenubarMsg>>>,
) {
    let Some(msg) = chan.into_inner().0.1.try_recv().unwrap() else {
        return;
    };
    let Some(ev) = menu.get_event(msg) else {
        return;
    };
    menu_tx.write(*ev);
}

#[derive(Message)]
struct OpenContentMsg;

fn handle_menubar_event(
    mut menu_rx: MessageReader<MainMenubarMsg>,
    mut open_content: MessageWriter<OpenContentMsg>,
    mut psx_cmdw: MessageWriter<PsxCmdMsg>,
    mut exit_tx: MessageWriter<AppExit>,
) {
    for msg in menu_rx.read() {
        match msg {
            MainMenubarMsg::OpenContent => {
                open_content.write(OpenContentMsg);
            }
            MainMenubarMsg::UnloadContent => {
                psx_cmdw.write(PsxCmdMsg::UnloadContent);
            }
            MainMenubarMsg::Quit => {
                exit_tx.write_default();
            }
        };
    }
}

enum AsyncTaskMsg {
    OpenContentReceivedPath(PathBuf),
}
#[derive(Resource, Deref)]
struct AsyncTaskChan(AsyncChan<AsyncTaskMsg>);

fn handle_app_events(
    mut open_content: MessageReader<OpenContentMsg>,
    task_chan: If<Res<AsyncTaskChan>>,
    ev_loop_proxy: If<Res<EventLoopProxyWrapper>>,
) {
    if !open_content.is_empty() {
        let pool = AsyncComputeTaskPool::get();
        let task_chan = task_chan.clone();
        let proxy = winit::EventLoopProxy::clone(&ev_loop_proxy);
        pool.spawn(async move {
            let file = AsyncFileDialog::new().pick_file().await;
            if let Some(file) = file {
                let path = file.path().to_owned();
                proxy.send_event(winit::WinitUserEvent::WakeUp).unwrap();
                task_chan
                    .0
                    .send(AsyncTaskMsg::OpenContentReceivedPath(path))
                    .await
                    .unwrap();
            }
        })
        .detach();

        open_content.clear();
    }
}

fn handle_async_task_msg(
    task_chan: If<Res<AsyncTaskChan>>,
    // mut psx: If<ResMut<Psx>>,
    mut open_content: MessageWriter<PsxCmdMsg>,
) -> Result<()> {
    while let Some(msg) = task_chan.1.try_recv()? {
        match msg {
            AsyncTaskMsg::OpenContentReceivedPath(path) => {
                open_content.write(PsxCmdMsg::OpenContentAt(path));
            }
        }
    }
    Ok(())
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
    app.add_message::<MainMenubarMsg>()
        .add_message::<OpenContentMsg>()
        .add_message::<UpdateMainMenubarMsg>()
        .add_systems(Update, (on_gamepad_connection, handle_gamepad_events))
        .add_systems(
            Update,
            (
                update_statusbar,
                update_main_menubar,
                handle_menubar_event.before(handle_app_events),
                handle_app_events.before(handle_async_task_msg),
                handle_async_task_msg,
            ),
        )
        .add_systems(PostUpdate, (on_window_event,))
        .add_systems(PreUpdate, (keyboard_shortcuts, menubar_channel_forward));
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
    mut exit_tx: MessageWriter<AppExit>,
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

    if cmd && input.just_pressed(KeyCode::KeyQ) {
        exit_tx.write_default();
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

fn on_gamepad_connection(
    mut reader: MessageReader<GamepadConnectionEvent>,
    mut psx: If<ResMut<Psx>>,
) {
    for conn in reader.read() {
        match &conn.connection {
            bevy::input::gamepad::GamepadConnection::Connected { name, .. } => {
                let first_vacant = Sio0Port::iter().find(|port| psx.gamepads[*port].is_none());
                if let Some(port) = first_vacant {
                    psx.gamepads[port] = Some(conn.gamepad);
                    psx.emu.sio.sio0ports.port_mut(port).joypad.plug_in();
                    tracing::info!("{name} connected on port {}.", port as u8);
                }
            }
            bevy::input::gamepad::GamepadConnection::Disconnected => {
                let connected_port = Sio0Port::iter()
                    .find(|port| psx.gamepads[*port].is_some_and(|port| port == conn.gamepad));
                if let Some(port) = connected_port {
                    psx.gamepads[port] = None;
                    psx.emu.sio.sio0ports.port_mut(port).joypad.plug_out();
                    tracing::info!("port {} disconnected.", port as u8);
                }
            }
        }
    }
}

fn handle_gamepad_events(gamepads: Query<&Gamepad>, mut psx: If<ResMut<Psx>>) -> Result<()> {
    for port in Sio0Port::iter() {
        let pad = &psx.gamepads[port];
        let Some(pad) = pad else {
            continue;
        };
        let gamepad = gamepads.get(*pad)?;

        for released in gamepad.get_just_released() {
            if let Some(btn) = gamepad_button_to_pchan_button(released) {
                psx.emu
                    .send_input_event(pchan_bind::input::InputEvent::Release(btn), port);
            }
        }

        for pressed in gamepad.get_just_pressed() {
            if let Some(btn) = gamepad_button_to_pchan_button(pressed) {
                tracing::info!(?btn);
                psx.emu
                    .send_input_event(pchan_bind::input::InputEvent::Press(btn), port);
            }
        }
    }

    Ok(())
}

fn gamepad_button_to_pchan_button(btn: &GamepadButton) -> Option<PchanButton> {
    match btn {
        GamepadButton::South => Some(PchanButton::X),
        GamepadButton::East => Some(PchanButton::Circle),
        GamepadButton::North => Some(PchanButton::Triangle),
        GamepadButton::West => Some(PchanButton::Square),
        GamepadButton::C => None,
        GamepadButton::Z => None,
        GamepadButton::LeftTrigger => Some(PchanButton::L1),
        GamepadButton::LeftTrigger2 => Some(PchanButton::L2),
        GamepadButton::RightTrigger => Some(PchanButton::R1),
        GamepadButton::RightTrigger2 => Some(PchanButton::R2),
        GamepadButton::Select => Some(PchanButton::Select),
        GamepadButton::Start => Some(PchanButton::Start),
        GamepadButton::Mode => None,
        GamepadButton::LeftThumb => Some(PchanButton::L3),
        GamepadButton::RightThumb => Some(PchanButton::R3),
        GamepadButton::DPadUp => Some(PchanButton::DpadUp),
        GamepadButton::DPadDown => Some(PchanButton::DpadDown),
        GamepadButton::DPadLeft => Some(PchanButton::DpadLeft),
        GamepadButton::DPadRight => Some(PchanButton::DpadRight),
        GamepadButton::Other(_) => None,
    }
}
