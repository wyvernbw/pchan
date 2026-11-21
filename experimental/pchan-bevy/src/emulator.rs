use core::mem;
use core::num::NonZeroU16;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use bevy::asset::RenderAssetUsages;

use bevy::core_pipeline::{Core2d, Core2dSystems};

use bevy::extract::extract_resource::ExtractResourcePlugin;

use bevy::image::ImageSampler;
use bevy::log::tracing;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;

use bevy::render::renderer::{RenderAdapter, RenderContext, RenderInstance, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{RenderApp, RenderStartup};
use bevy::tasks::AsyncComputeTaskPool;
use bevy::ui::UiSystems;
use bevy::window::PrimaryWindow;
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::io::cdrom::OpenDiscState;
use pchan_emu::io::sio::Sio0Port;
use pchan_emu::run::{EmuSpeed, Runner, RunnerConfig, RunnerMode};
use pchan_gpu::wgpu;
use pchan_utils::{Indexed, kanal};

use crate::ui::RerenderMsg;
use crate::{UpdateMainMenubarMsg, app_title};

pub fn emulator_plugin(app: &mut App) {
    let hwr_chan = HwrChannel(kanal::bounded_async(1));

    app.init_state::<PsxState>()
        .init_state::<HwrState>()
        .init_state::<PsxLoopKindState>()
        .add_plugins(ExtractResourcePlugin::<DisplayTarget, RenderApp>::default())
        .insert_resource(Time::from_hz(59.826)) // NTSC
        .insert_resource::<HwrChannel>(hwr_chan.clone())
        .init_resource::<DisplaySize>()
        .add_systems(Update, try_recv_hwr_chan)
        .add_systems(OnEnter(HwrState::Ready), setup_psx)
        .add_systems(
            FixedUpdate,
            psx_loop.run_if(
                in_state(PsxState::Running)
                    .or_else(in_state(PsxState::RunOneFrame))
                    .and_then(in_state(PsxLoopKindState::FixedStep)),
            ),
        )
        .add_systems(
            Update,
            psx_loop.run_if(
                in_state(PsxState::Running)
                    .or_else(in_state(PsxState::RunOneFrame))
                    .and_then(in_state(PsxLoopKindState::HotLoop)),
            ),
        )
        .add_systems(Update, track_display_size.after(UiSystems::Layout))
        .add_message::<PsxCmdMsg>()
        .add_systems(Update, emu_handle_open_content_at);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .insert_resource(hwr_chan)
            .add_systems(RenderStartup, setup_hwr)
            .add_systems(Core2d, display_pass.in_set(Core2dSystems::Prepass));
    } else {
        panic!("renderer not set up.")
    }
}

#[derive(Resource)]
pub struct Psx {
    pub emu:      Emu,
    pub runner:   Runner,
    pub elapsed:  VecDeque<f32>,
    pub gamepads: Indexed<[Option<Entity>; 2], Sio0Port>,
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
enum HwrState {
    #[default]
    SettingUp,
    Ready,
}

#[derive(Resource)]
pub struct Hwr {
    pub renderer: Arc<pchan_gpu::Renderer>,
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
    mut cmds: Commands,
) {
    tracing::info!("setting up hw renderer.");
    let wgpu = pchan_gpu::WgpuBackend {
        instance:       instance.to_owned().into_inner(),
        adapter:        adapter.to_owned().into_inner(),
        device:         ctx.render_device().wgpu_device().clone(),
        queue:          queue.to_owned().into_inner(),
        display_format: pchan_gpu::wgpu::TextureFormat::Rgba8UnormSrgb,
    };
    let rd = pchan_gpu::Renderer::from_wgpu(wgpu).expect("faield to create hwr.");
    let mut dp = rd.display_uniforms.app.lock().unwrap();
    dp.screen_rect.x = 320;
    dp.screen_rect.y = 240;
    drop(dp);

    let rd = Arc::new(rd);
    rd.clone().start();

    cmds.insert_resource(Hwr {
        renderer: rd.clone(),
    });

    hwr_chan.0.0.as_sync().send(Hwr { renderer: rd }).unwrap();
}

pub const EMULATOR_ELAPSED_RINGBUF_SIZE: usize = 240;

fn setup_psx(
    mut cmds: Commands,
    hwr: Res<Hwr>,
    mut images: ResMut<Assets<Image>>,
    mut redraw: MessageWriter<RerenderMsg>,
) -> Result<()> {
    tracing::info!("setting up psx loop.");

    let mut emu = Emu::new();
    emu.set_bios_path(std::env::var("PCHAN_BIOS")?);
    emu.load_bios()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();

    let mut audio_task = AudioTask::new()?;
    pchan_bind::bind_audio(&mut audio_task, &mut emu);
    let audio_stream = audio_task.start()?;
    mem::forget(audio_stream);

    hwr.renderer.connect_emu(&mut emu);

    let mut runner = Runner::new().with_config(RunnerConfig {
        force_mode: Some(RunnerMode::Dynarec),
        speed:      EmuSpeed::Percentage(NonZeroU16::new(100).unwrap()),
    });
    runner.running = false;

    cmds.set_state(match runner.running {
        true => PsxState::Running,
        false => PsxState::Paused,
    });
    cmds.insert_resource(Psx {
        emu,
        runner,
        elapsed: VecDeque::with_capacity(EMULATOR_ELAPSED_RINGBUF_SIZE),
        gamepads: Indexed::from([None; 2]),
    });

    let mut image = Image::new_uninit(
        wgpu::Extent3d {
            width:                 640,
            height:                480,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let usage = wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::RENDER_ATTACHMENT;
    image.texture_descriptor = wgpu::TextureDescriptor {
        mip_level_count: 1,
        sample_count: 1,
        usage,
        ..image.texture_descriptor
    };
    image.sampler = ImageSampler::nearest();
    let handle = images.add(image);
    cmds.insert_resource(DisplayTarget(handle));

    redraw.write(RerenderMsg);

    Ok(())
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
enum PsxLoopKindState {
    #[default]
    FixedStep,
    HotLoop,
}

fn psx_loop(
    mut cmds: Commands,
    mut psx: ResMut<Psx>,
    state: Res<State<PsxState>>,
    mut ev: MessageWriter<RerenderMsg>,
) {
    let state = *state.get();
    match state {
        PsxState::Setup | PsxState::Paused => {}
        PsxState::Running | PsxState::RunOneFrame => {
            let pool = AsyncComputeTaskPool::get();
            let start = Instant::now();
            pool.scope(|scope| {
                scope.spawn(async {
                    let Psx { emu, runner, .. } = &mut *psx;
                    runner.run_until_vblank(emu).await;
                    ev.write(RerenderMsg);
                });
            });
            let elapsed = start.elapsed().as_millis_f32();
            if psx.elapsed.len() == EMULATOR_ELAPSED_RINGBUF_SIZE {
                psx.elapsed.pop_front();
            }
            psx.elapsed.push_back(elapsed);

            if state == PsxState::RunOneFrame {
                psx.pause(&mut cmds);
            }
        }
    }
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PsxState {
    #[default]
    Setup,
    Paused,
    Running,
    RunOneFrame,
}

#[derive(Component, Clone, Copy, Default)]
pub struct DisplayNode;

#[derive(Resource, Default)]
struct DisplaySize(UVec2);

fn track_display_size(
    node: If<Single<&ComputedNode, With<DisplayNode>>>,
    target: Res<DisplayTarget>,
    mut size_res: ResMut<DisplaySize>,
    mut images: ResMut<Assets<Image>>,
) {
    let new_size = node.size().as_uvec2();
    if new_size.x != 0
        && new_size.y != 0
        && new_size != size_res.0
        && let Some(mut img) = images.get_mut(&target.0)
    {
        img.resize(wgpu::Extent3d {
            width:                 new_size.x,
            height:                new_size.y,
            depth_or_array_layers: 1,
        });
    }
    *size_res = DisplaySize(new_size);
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct DisplayTarget(pub Handle<Image>);

fn display_pass(
    target: If<Res<DisplayTarget>>,
    images: Res<RenderAssets<GpuImage>>,
    hwr: If<Res<Hwr>>,
    mut ctx: RenderContext,
) {
    let Some(img) = images.get(&target.into_inner().0) else {
        return;
    };

    {
        let mut dp = hwr.renderer.display_uniforms.app.lock().unwrap();
        let size = img.size_2d();
        dp.screen_rect.x = size.x as u16;
        dp.screen_rect.y = size.y as u16;

        if size.x == 0 || size.y == 0 {
            return;
        };
    }

    let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("display"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view:           &img.texture_view,
            depth_slice:    None,
            resolve_target: None,
            ops:            wgpu::Operations {
                load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        ..default()
    });

    hwr.renderer.draw_display(pass.wgpu_pass());
}

impl Psx {
    pub fn toggle_running(&mut self, cmds: &mut Commands) {
        self.runner.running.toggle();
        let state = match self.runner.running {
            true => PsxState::Running,
            false => PsxState::Paused,
        };
        cmds.set_state_if_different(state);
    }
    pub fn run_one_frame(&mut self, cmds: &mut Commands) {
        self.runner.running = true;
        cmds.set_state(PsxState::RunOneFrame);
    }
    pub fn pause(&mut self, cmds: &mut Commands) {
        self.runner.running = false;
        cmds.set_state(PsxState::Paused);
    }

    pub fn open_content(&mut self, path: &Path) -> Result<()> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("exe") => {
                let exe = fs::read(path)?;
                self.emu.sideload_exe(&exe)?;
            }
            Some("cue" | "CUE") => {
                let cue_content = fs::read_to_string(path)?;
                let mut fsm = self.emu.open_disc_with_cue(&cue_content, true)?;
                let root = path.parent().unwrap_or(Path::new(""));
                loop {
                    match fsm {
                        OpenDiscState::Done(fsm) => {
                            fsm.commit(&mut self.emu);
                            break;
                        }
                        OpenDiscState::Opening(next) => {
                            fsm = next.advance()?;
                        }
                        OpenDiscState::NeedCue(next) => {
                            fsm = next.advance(&cue_content)?;
                        }
                        OpenDiscState::NeedFile(next) => {
                            let path = root.join(next.requested_filename);
                            tracing::info!("opening {}", path.display());

                            let file = fs::File::open(path)?;
                            let file = Box::new(file);
                            fsm = OpenDiscState::Opening(next.advance(file as _)?);
                        }
                    }
                }
                tracing::info!("opened: {:#?}", self.emu.cdrom.cue_sheet());
            }
            _ => {
                let file = fs::File::open(path)?;
                let file = Box::new(file);
                self.emu.open_disc_raw(file, true)?;
            }
        }
        Ok(())
    }

    pub fn has_content(&self) -> bool {
        self.emu.cdrom.disc().is_some() || self.emu.sideloaded_exe().is_some()
    }
}

#[derive(Message)]
pub enum PsxCmdMsg {
    OpenContentAt(PathBuf),
    UnloadContent,
    SetSpeed(EmuSpeed),
}

fn emu_handle_open_content_at(
    mut cmds: Commands,
    mut reader: MessageReader<PsxCmdMsg>,
    mut psx: If<ResMut<Psx>>,
    mut win: Option<Single<&mut Window, With<PrimaryWindow>>>,
    mut menubarw: Option<MessageWriter<UpdateMainMenubarMsg>>,
    mut time: If<ResMut<Time<Fixed>>>,
) -> Result<()> {
    let ntsc60fps = 59.826;

    for cmd in reader.read() {
        match cmd {
            PsxCmdMsg::OpenContentAt(path) => {
                psx.open_content(path)?;
                if let Some(win) = &mut win {
                    let playing = path.file_name().and_then(|f| f.to_str());
                    app_title(&mut win.title, playing);
                }
                if let Some(menubarw) = &mut menubarw {
                    menubarw.write_default();
                }
            }
            PsxCmdMsg::UnloadContent => {
                let _ = psx.emu.cdrom.take_disc();
                psx.emu.boot.remove_sideload();
                if let Some(menubarw) = &mut menubarw {
                    menubarw.write_default();
                }
            }
            PsxCmdMsg::SetSpeed(emu_speed) => {
                tracing::info!("set speed: {emu_speed:?}");
                psx.runner.config.speed = *emu_speed;
                match emu_speed {
                    EmuSpeed::Unlimited => {
                        cmds.set_state(PsxLoopKindState::HotLoop);
                    }
                    EmuSpeed::Percentage(speed) => {
                        cmds.set_state(PsxLoopKindState::FixedStep);
                        **time = Time::from_hz(ntsc60fps * f64::from(speed.get()) / 100.);
                    }
                }

                if let Some(win) = &mut win {
                    if emu_speed.is_unlimited() || emu_speed.is_sped_up() {
                        win.present_mode = bevy::window::PresentMode::AutoNoVsync;
                    } else {
                        win.present_mode = bevy::window::PresentMode::AutoVsync;
                    }
                }
            }
        }
    }
    Ok(())
}
