use core::mem;
use core::num::NonZeroU16;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use alloc::alloc::Global;
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
use bevy::window::{PrimaryWindow, RequestRedraw};
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::run::{EmuSpeed, Runner, RunnerConfig, RunnerMode};
use pchan_gpu::wgpu;
use pchan_utils::kanal;

use crate::app_title;

pub fn emulator_plugin(app: &mut App) {
    let hwr_chan = HwrChannel(kanal::bounded_async(1));

    app.init_state::<PsxState>()
        .init_state::<HwrState>()
        .add_plugins(ExtractResourcePlugin::<DisplayTarget, RenderApp>::default())
        .insert_resource(Time::from_hz(59.826)) // NTSC
        .insert_resource::<HwrChannel>(hwr_chan.clone())
        .init_resource::<DisplaySize>()
        .add_systems(Update, try_recv_hwr_chan)
        .add_systems(OnEnter(HwrState::Ready), setup_psx)
        .add_systems(
            FixedUpdate,
            psx_loop.run_if(in_state(PsxState::Running).or_else(in_state(PsxState::RunOneFrame))),
        )
        .add_systems(Update, track_display_size.after(UiSystems::Layout))
        .add_message::<OpenContentAt>()
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
    pub emu:     Emu,
    pub runner:  Runner<Global>,
    pub elapsed: VecDeque<f32>,
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

fn setup_psx(mut cmds: Commands, hwr: Res<Hwr>, mut images: ResMut<Assets<Image>>) -> Result<()> {
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
    runner.running = false;

    cmds.set_state(match runner.running {
        true => PsxState::Running,
        false => PsxState::Paused,
    });
    cmds.insert_resource(Psx {
        emu,
        runner,
        elapsed: VecDeque::with_capacity(EMULATOR_ELAPSED_RINGBUF_SIZE),
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

    Ok(())
}

fn psx_loop(
    mut cmds: Commands,
    mut psx: ResMut<Psx>,
    state: Res<State<PsxState>>,
    mut ev: MessageWriter<RequestRedraw>,
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
                    ev.write(RequestRedraw);
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
                self.emu.sideload_exe(&exe, self.emu.alloc)?;
            }
            _ => {
                let fsm = self.emu.open_disc(path, true)?;
                self.emu.advance_open_disc(path, fsm, true)?;
            }
        }
        Ok(())
    }
}

#[derive(Message)]
pub struct OpenContentAt(pub PathBuf);

fn emu_handle_open_content_at(
    mut reader: MessageReader<OpenContentAt>,
    mut psx: If<ResMut<Psx>>,
    mut win: Option<Single<&mut Window, With<PrimaryWindow>>>,
) -> Result<()> {
    for OpenContentAt(path) in reader.read() {
        psx.open_content(path)?;
        if let Some(win) = &mut win {
            let playing = path.file_name().and_then(|f| f.to_str());
            app_title(&mut win.title, playing);
        }
    }
    Ok(())
}
