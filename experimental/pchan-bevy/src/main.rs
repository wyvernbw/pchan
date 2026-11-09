#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

use core::mem;
use core::num::NonZeroU16;
use std::sync::Mutex;

use alloc::alloc::Global;
use alloc::sync::Arc;
use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::*;
use bevy::log::tracing;
use bevy::prelude::*;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderInstance, RenderQueue};
use bevy::render::{RenderApp, RenderStartup};
use bevy::tasks::AsyncComputeTaskPool;
use bevy::window::RequestRedraw;
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::run::{EmuSpeed, Runner, RunnerConfig, RunnerMode};
use pchan_utils::kanal;

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
        .add_plugins(emulator_plugin)
        .add_plugins(app_ui_plugin)
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

pub fn emulator_plugin(app: &mut App) {
    let hwr_chan = HwrChannel(kanal::bounded_async(1));

    app.init_state::<PsxState>()
        .init_state::<HwrState>()
        .insert_resource(Time::from_hz(59.826)) // NTSC
        .insert_resource::<HwrChannel>(hwr_chan.clone())
        .add_systems(Update, try_recv_hwr_chan)
        .add_systems(OnEnter(HwrState::Ready), setup_psx)
        .add_systems(FixedUpdate, psx_loop.run_if(in_state(PsxState::Running)));

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .insert_resource(hwr_chan)
            .add_systems(RenderStartup, setup_hwr);
    } else {
        panic!("renderer not set up.")
    }
}

#[derive(Resource)]
struct Psx {
    emu:    Emu,
    runner: Runner<Global>,
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
enum HwrState {
    #[default]
    SettingUp,
    Ready,
}

#[derive(Resource)]
struct Hwr {
    renderer: Arc<pchan_gpu::Renderer>,
}
#[derive(Resource, Clone)]
struct HwrChannel(pchan_utils::AsyncChan<Hwr>);

fn try_recv_hwr_chan(hwr_chan: Res<HwrChannel>, mut cmds: Commands) {
    if let Some(hwr) = hwr_chan.0.1.as_sync().try_recv().unwrap() {
        cmds.insert_resource(hwr);
        cmds.set_state(HwrState::Ready);
    }
}

fn setup_hwr(
    hwr_chan: Res<HwrChannel>,
    ctx: RenderContext,
    queue: Res<RenderQueue>,
    instance: Res<RenderInstance>,
    adapter: Res<RenderAdapter>,
) {
    tracing::info!("setting up hw renderer.");
    let wgpu = pchan_gpu::WgpuBackend {
        instance:       instance.to_owned().into_inner(),
        adapter:        adapter.to_owned().into_inner(),
        device:         ctx.render_device().wgpu_device().clone(),
        queue:          queue.to_owned().into_inner(),
        display_format: pchan_gpu::wgpu::TextureFormat::Bgra8UnormSrgb,
    };
    let rd = pchan_gpu::Renderer::from_wgpu(wgpu).expect("faield to create hwr.");
    let mut dp = rd.display_uniforms.app.lock().unwrap();
    dp.screen_rect.x = 320;
    dp.screen_rect.y = 240;
    drop(dp);

    let rd = Arc::new(rd);
    rd.clone().start();

    hwr_chan.0.0.as_sync().send(Hwr { renderer: rd }).unwrap();
}

fn setup_psx(mut cmds: Commands, hwr: Res<Hwr>) -> Result<()> {
    tracing::info!("setting up psx loop.");

    let mut emu = Emu::new();
    emu.set_bios_path(std::env::var("PCHAN_BIOS")?);
    emu.load_bios(Global)?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();

    let mut audio_task = AudioTask::new()?;
    pchan_bind::bind_audio(&mut audio_task, &mut emu);
    let audio_stream = audio_task.start()?;
    mem::forget(audio_stream);

    hwr.renderer.connect_emu(&mut emu);

    let mut runner = Runner::new_in(Global).with_config(RunnerConfig {
        force_mode: Some(RunnerMode::Dynarec),
        speed:      EmuSpeed::Percentage(NonZeroU16::new(100).unwrap()),
    });
    // runner.running = false;

    cmds.insert_resource(Psx { emu, runner });
    cmds.set_state(PsxState::Running);

    Ok(())
}

fn psx_loop(mut psx: ResMut<Psx>, mut ev: MessageWriter<RequestRedraw>) {
    let pool = AsyncComputeTaskPool::get();
    pool.scope(|scope| {
        scope.spawn(async {
            let Psx { emu, runner } = &mut *psx;
            runner.run_until_vblank(emu).await;
            ev.write(RequestRedraw);
        });
    });
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
enum PsxState {
    #[default]
    Setup,
    Paused,
    Running,
}

#[derive(Component, Clone, Default)]
struct StatusbarFrameCounter;

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
            StatusbarFrameCounter
            @text("#frame")
        ]
    }
}

fn app_ui_plugin(app: &mut App) {
    app.add_systems(Update, update_statusbar);
}

fn update_statusbar(psx: If<Res<Psx>>, frame_cnt: Query<&mut Text, With<StatusbarFrameCounter>>) {
    use core::fmt::Write;

    for mut text in frame_cnt {
        text.0.clear();
        _ = write!(text.0, "#{}", psx.runner.frame_idx);
    }
}

fn text(value: impl AsRef<str>) -> impl Scene {
    let value = value.as_ref();
    bsn! {
        Text(value)
        TextFont {
            font: bevy::text::FontSourceTemplate::Generic(GenericFontFamily::SansSerif),
            font_size: rem(1),
        }
    }
}
