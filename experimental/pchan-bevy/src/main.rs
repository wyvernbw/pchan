#![feature(duration_millis_float)]
#![allow(recursion_depth_exceeding_limit)]
#![allow(clippy::too_many_arguments)]

extern crate alloc;

use core::marker::PhantomData;
use core::num::NonZero;
use std::collections::HashMap;
use std::path::PathBuf;

use bevy::camera::RenderTarget;
use bevy::ecs::system::NonSendMarker;
use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::*;
use bevy::input::mouse::MouseScrollPixelsPerLine;
use bevy::log::{LogPlugin, tracing};
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use bevy::ui::ui_layout_system;
use bevy::window::{PrimaryWindow, WindowEvent, WindowRef};
use bevy::winit::EventLoopProxyWrapper;
#[cfg(target_os = "macos")]
use bevy::winit::WINIT_WINDOWS;
use bevy::{window, winit};
use muda::accelerator::{Accelerator, Code, Modifiers};
use muda::{CheckMenuItem, IsMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};
use pchan_emu::run::EmuSpeed;
use pchan_utils::{AsyncChan, Chan, kanal};
use rfd::AsyncFileDialog;

mod emulator;
mod ui;

use crate::emulator::{DisplayNode, DisplayTarget, Psx, PsxCmdMsg, PsxState, emulator_plugin};
use crate::ui::{RerenderMsg, Theme, UiCamera, text, ui_plugin, vseparator};

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
                        title,
                        ..new_window()
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

#[must_use]
pub fn new_window() -> Window {
    Window {
        titlebar_shown: true,
        titlebar_transparent: true,
        fullsize_content_view: true,
        ..default()
    }
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
    ViewMemory,
    Quit,
    SetSpeed(EmuSpeed),
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
    fn menu_item<I: IsMenuItem>(&mut self, item: I, ev: Ev) -> I {
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

fn build_main_menubar(
    psx: &Psx,
    memview_window: Option<&Single<Entity, With<MemoryViewerWindow>>>,
) -> NativeMenubar<MainMenubarMsg> {
    fn speed_option(
        psx: &Psx,
        menu: &mut NativeMenubar<MainMenubarMsg>,
        label: &str,
        speed: EmuSpeed,
    ) -> CheckMenuItem {
        let speed_item_checked = psx.runner.config().speed == speed;

        menu.menu_item(
            CheckMenuItem::new(label, true, speed_item_checked, None),
            MainMenubarMsg::SetSpeed(speed),
        )
    }

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

    let emu_speed_100 = EmuSpeed::Percentage(NonZero::new(100).unwrap());
    let speed100 = speed_option(psx, &mut menu, "100%", emu_speed_100);

    let emu_speed_200 = EmuSpeed::Percentage(NonZero::new(200).unwrap());
    let speed200 = speed_option(psx, &mut menu, "200%", emu_speed_200);

    let speed_unlimited = speed_option(psx, &mut menu, "Unlimited", EmuSpeed::Unlimited);

    let set_speed =
        Submenu::with_items("Speed", true, &[&speed100, &speed200, &speed_unlimited]).unwrap();

    let view_memory = menu.menu_item(
        CheckMenuItem::new("View Memory", true, memview_window.is_some(), None),
        MainMenubarMsg::ViewMemory,
    );

    let debug_menu = Submenu::with_items(
        "Debug",
        true,
        &[&set_speed, &PredefinedMenuItem::separator(), &view_memory],
    )
    .unwrap();
    menu.append(&debug_menu).unwrap();

    menu
}

fn setup(world: &mut World) -> Result<()> {
    let display = world.resource::<DisplayTarget>();
    let theme = world.resource::<Theme>();
    world
        .spawn_scene_list(bsn_list! {
             Camera2d
             UiCamera
             --
             @root(&theme, &display.0)
        })
        .unwrap();

    world.write_message(RerenderMsg);

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

    let menu = NativeMenubar::<MainMenubarMsg>::new();
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

    world.write_message_default::<UpdateMainMenubarMsg>();
    world.write_message(RerenderMsg);

    Ok(())
}

#[derive(Message, Clone, Copy, Default)]
pub struct UpdateMainMenubarMsg;

fn update_main_menubar(
    mut menubar: If<NonSendMut<NativeMenubar<MainMenubarMsg>>>,
    psx: If<Res<Psx>>,
    mut reader: MessageReader<UpdateMainMenubarMsg>,
    win: If<Single<&window::RawHandleWrapper, With<PrimaryWindow>>>,
    memview_window: Option<Single<Entity, With<MemoryViewerWindow>>>,
) {
    for _ in reader.read() {
        **menubar = build_main_menubar(&psx, memview_window.as_ref());
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

#[derive(Component, Clone, Copy, Default)]
struct MemoryViewerWindow;

fn handle_menubar_event(
    mut cmds: Commands,
    mut menu_rx: MessageReader<MainMenubarMsg>,
    mut open_content: MessageWriter<OpenContentMsg>,
    mut psx_cmdw: MessageWriter<PsxCmdMsg>,
    mut exit_tx: MessageWriter<AppExit>,
    mut psx: If<ResMut<Psx>>,
    theme: Res<Theme>,
    memview_window: Option<Single<Entity, With<MemoryViewerWindow>>>,
) {
    for msg in menu_rx.read() {
        match msg {
            MainMenubarMsg::OpenContent => {
                open_content.write(OpenContentMsg);
            }
            MainMenubarMsg::UnloadContent => {
                psx_cmdw.write(PsxCmdMsg::UnloadContent);
            }
            MainMenubarMsg::SetSpeed(emu_speed) => {
                psx_cmdw.write(PsxCmdMsg::SetSpeed(*emu_speed));
            }
            MainMenubarMsg::Quit => {
                psx.emu.dma.dump_cdrom_data();
                psx.emu.cdrom.dump_audio_data();
                exit_tx.write_default();
            }
            MainMenubarMsg::ViewMemory => match &memview_window {
                Some(entity) => {
                    cmds.entity(**entity).despawn();
                }
                None => {
                    let win = cmds.spawn((new_window(), MemoryViewerWindow)).id();
                    let camera = cmds
                        .spawn((
                            Camera2d,
                            UiCamera,
                            RenderTarget::Window(WindowRef::Entity(win)),
                        ))
                        .id();
                    cmds.spawn_scene(bsn! {
                        @memview_root(&theme, &psx)
                        UiTargetCamera(camera)
                    });
                }
            },
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

#[must_use]
pub fn titlebar() -> Node {
    Node {
        height: rem(1.75),
        width: percent(100),
        padding: UiRect::horizontal(rem(1)),
        ..default()
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
            titlebar()
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
        .add_systems(PreUpdate, (keyboard_shortcuts, menubar_channel_forward))
        .add_plugins(virtual_list_plugin)
        .register_list_type::<MemviewList>();
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
    mut ev: MessageWriter<RerenderMsg>,
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
        psx.emu.dma.dump_cdrom_data();
        exit_tx.write_default();
    }

    if psx_state == PsxState::Paused && cmd && input.pressed(KeyCode::KeyK) {
        psx.run_one_frame(&mut cmds);
        ev.write(RerenderMsg);
    }
}

#[derive(Component, Default, Clone, Copy)]
struct VirtualListState {
    first:        usize,
    last:         usize,
    scroll_fract: f32,
}
#[derive(Component, Default, Clone, Copy, Deref, DerefMut)]
#[require(VirtualListState, VirtualScrollPosition)]
struct VirtualListContainer<T>(T);
#[derive(Component, Default, Clone, Copy)]
struct VirtualListInner<T>(PhantomData<T>);

#[derive(Component, Default, Clone, Copy, Deref, DerefMut, Debug)]
struct VirtualScrollPosition(f32);

#[derive(Component, Default, Clone, Copy)]
struct MemviewList;

pub fn memview_root(theme: &Theme, psx: &Psx) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            padding: rem(0.5),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            position_type: PositionType::Absolute
        }
        BackgroundColor({theme.bg})
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            titlebar()
            --
            Node {
                height: percent(50),
                max_height: percent(50),
                min_height: px(0),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                overflow: Overflow::hidden(),
                border: UiRect::all(px(2))
            }
            BorderColor::all(Color::srgb(0.0, 0.5, 1.0))
            VirtualListContainer<MemviewList>::default()
            Children [
                VirtualListInner<MemviewList>::default()
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    border: UiRect::all(px(2))
                }
                BorderColor::all(Color::srgb(0.75, 0.25, 0.))
            ]
        ]
    }
}

impl VirtualList for MemviewList {
    fn get_row(&mut self, index: usize) -> Option<impl Scene> {
        Some(bsn! {
            @text(format!("hello world! #{index}"))
        })
    }

    fn height(&self) -> Val {
        px(14.)
    }
}

pub trait VirtualListPluginExt {
    fn register_list_type<T: VirtualList>(&mut self);
}

impl VirtualListPluginExt for App {
    fn register_list_type<T: VirtualList>(&mut self) {
        self.add_systems(PostUpdate, virtual_list_init::<T>.after(ui_layout_system))
            .add_observer(virtual_list_update_rows::<T>);
    }
}

impl VirtualListState {
    fn len(&self) -> usize {
        self.last.saturating_sub(self.first)
    }
}

pub trait VirtualList: Component {
    fn get_row(&mut self, index: usize) -> Option<impl Scene>;
    fn height(&self) -> Val;
}

fn virtual_list_plugin(app: &mut App) {}

fn virtual_list_init<T: VirtualList>(
    mut cmds: Commands,
    query_containers: Query<
        (
            &ComputedNode,
            &mut VirtualListState,
            &Children,
            &mut VirtualListContainer<T>,
        ),
        Added<VirtualListContainer<T>>,
    >,
    query_inner: Query<Entity, With<VirtualListInner<T>>>,
) -> Result<()> {
    for (container, mut state, children, mut list) in query_containers {
        let inner = children.iter().find_map(|c| query_inner.get(c).ok());
        let Some(inner_entity) = inner else {
            continue;
        };

        let row_height = list.height().resolve(
            container.inverse_scale_factor.recip(),
            container.size.y,
            container.size,
            container.em_size,
            container.rem_size,
        )?;
        let row_count = (container.size.y / row_height).floor() as usize;
        dbg!(container.size.y);

        let old = state.last;
        state.last += row_count;
        for idx in old..state.last {
            let Some(row) = list.0.get_row(idx) else {
                break;
            };
            let scene = cmds.spawn_scene(row).id();
            cmds.entity(inner_entity).add_child(scene);
        }
        dbg!(row_count);
    }
    Ok(())
}

fn virtual_list_update_rows<T: VirtualList>(
    scroll: On<PointerScroll>,
    mut cmds: Commands,
    mut query_containers: Query<(
        &ComputedNode,
        &mut VirtualScrollPosition,
        &mut VirtualListState,
        &Children,
        &mut VirtualListContainer<T>,
    )>,
    mut query_inner: Query<(Entity, &mut Node, &Children), With<VirtualListInner<T>>>,
    scroll_conversion_ratio: Res<MouseScrollPixelsPerLine>,
) -> Result<()> {
    if let Ok((container, mut scroll_position, mut state, children, mut list)) =
        query_containers.get_mut(scroll.entity)
    {
        let row_height = list.height().resolve(
            container.inverse_scale_factor.recip(),
            container.size.y,
            container.size,
            container.em_size,
            container.rem_size,
        )?;
        let row_height_unscaled = row_height * container.inverse_scale_factor;

        let entity = children
            .iter()
            .find(|c| query_inner.contains(*c))
            .expect("virtual list container requires an inner element.");
        let (inner_entity, mut inner_node, inner_children) = query_inner.get_mut(entity)?;

        let scroll_delta = scroll.to_pixels(&scroll_conversion_ratio).y;
        **scroll_position += scroll_delta;
        state.scroll_fract += scroll_delta.abs();
        eprintln!("{:?}", (**scroll_position, state.scroll_fract, row_height));

        inner_node.top = px(**scroll_position);

        let total_height = row_height * inner_children.len() as f32;
        let new_row_count = (state.scroll_fract / row_height_unscaled).floor() as usize;
        state.scroll_fract -= new_row_count as f32 * row_height_unscaled;

        // dbg!(state.scroll_fract / row_height);

        let old = state.last;
        state.last += new_row_count;
        for idx in old..state.last {
            let Some(row) = list.0.get_row(idx) else {
                break;
            };
            let scene = cmds.spawn_scene(row).id();
            cmds.entity(inner_entity).add_child(scene);
        }
    }
    Ok(())
}
