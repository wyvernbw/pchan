#![feature(duration_millis_float)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(dead_code)]
#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

mod frontend;
mod render_pass;
mod utils;
mod vertex;

use alloc::slice;
use arbitrary_int::prelude::*;
use blink_alloc::BlinkAlloc;
pub use glam;
use pchan_emu::memory::mb;
use pchan_utils::default;
use pchan_utils::tracy::TracyClient;

use alloc::sync::Arc;
use core::alloc::Allocator;
use core::num::NonZeroU64;
use core::sync::atomic::AtomicBool;
use core::{mem, ptr};
use std::sync::Mutex;
use std::time::Instant;
use wgpu::util::StagingBelt;

use glam::prelude::*;
use pchan_emu::Emu;
use pchan_emu::gpu::draw_call::{Gp0HostBlit, Gp0VramBlit};
use pchan_emu::gpu::{
    Conn, Display, DisplayColorDepth, DrawCallMsg, FlushMsg, GpuStatReg, VramSink,
};
use thiserror::Error;
pub use wgpu;
use wgpu::*;

use crate::frontend::{CPU_TO_VRAM_UNIFORM_BUF_SIZE, Pass};
use crate::vertex::Vertex;

#[derive(Debug)]
pub struct Renderer {
    pub gpu: WgpuBackend,
    pub res: BackendResources,
    pub display_uniforms: DisplayUniforms,
    pub tracy: TracyClient,
    clear_display_flag: AtomicBool,

    conn: Conn,
}

#[derive(Debug)]
pub struct WgpuBackend {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pub display_format: TextureFormat,
}

#[derive(Debug)]
pub struct BackendResources {
    render_pipelines: RenderPipelines,
    pub display_pipeline: RenderPipeline,
    pub vertex_buf: Buffer,
    pub render_texture: Texture,
    pub render_view: TextureView,
    render_bind_group: BindGroup,
    vram_texture: Texture,
    vram_out_buf: Buffer,
    vram_blit_buffer: Buffer,
    pub display_bind_group: BindGroup,
    pub display_uniform_buffer: Buffer,
    belt: Mutex<StagingBelt>,
    blit: BlitV2VResources,
    readback: ReadbackResources,
}

#[derive(Debug)]
struct RenderPipelines {
    add: RenderPipeline,
    subtract: RenderPipeline,
    subtract_opaque: RenderPipeline,
}

const VRAM_FMT: TextureFormat = TextureFormat::Rgba8Unorm;

/// Uniforms passed to display shader
#[derive(Debug)]
pub struct DisplayUniforms {
    pub hw: Mutex<HWDisplayUniforms>,
    pub app: Mutex<AppDisplayUniforms>,
}

/// Display uniforms relating to hardware settings
#[derive(Debug, Clone, Default)]
pub struct HWDisplayUniforms {
    pub dp_start: U16Vec2,
    pub dp_res: U16Vec2,
    pub dp_color_depth: DisplayColorDepth,
}

impl HWDisplayUniforms {
    #[must_use]
    pub fn from_hw(display: &Display, gpustat: GpuStatReg) -> Self {
        HWDisplayUniforms {
            dp_res: gpustat.resolution(),
            dp_start: display.display_vram_start,
            dp_color_depth: gpustat.display_color_depth(),
        }
    }
}

/// Display uniforms relating to application settings
#[derive(Debug, Clone, Default)]
pub struct AppDisplayUniforms {
    pub screen_rect: U16Vec2,
    pub dp_debug: bool,
    pub dp_srgb: bool,
}

impl Default for DisplayUniforms {
    fn default() -> Self {
        Self {
            hw: Mutex::new(HWDisplayUniforms {
                dp_start: U16Vec2::default(),
                dp_res: U16Vec2::default(),
                dp_color_depth: DisplayColorDepth::Depth15Bit,
            }),
            app: Mutex::new(AppDisplayUniforms {
                screen_rect: U16Vec2::default(),
                dp_debug: Default::default(),
                dp_srgb: true,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderUniforms {}

#[derive(Debug, Error)]
pub enum InitError {
    #[error(transparent)]
    RequestAdapter(#[from] RequestAdapterError),
    #[error(transparent)]
    RequestDevice(#[from] RequestDeviceError),
}

impl Renderer {
    fn create_render_pipeline(
        device: &Device,
        pipeline_layout: &PipelineLayout,
        shader_module: &ShaderModule,
        blend_state: Option<BlendState>,
    ) -> RenderPipeline {
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(pipeline_layout),
            vertex: VertexState {
                module: shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64, // 8
                    step_mode: VertexStepMode::Vertex,
                    attributes: Vertex::desc(),
                })],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                // psx gpu does not perform backface/frontface culling
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: VRAM_FMT,
                    blend: blend_state,
                    write_mask: ColorWrites::default(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    pub fn from_wgpu(
        WgpuBackend {
            instance,
            adapter,
            device,
            queue,
            display_format,
        }: WgpuBackend,
    ) -> Result<Self, InitError> {
        let shader_module = device.create_shader_module(include_wgsl!("../shaders/psx-gp0.wgsl"));
        let render_texture = device.create_texture(&TextureDescriptor {
            label: Some("pchan_gpu::render_tex"),
            size: Extent3d {
                width: 1024,
                height: 512,
                ..Default::default()
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: VRAM_FMT,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[VRAM_FMT],
        });
        let render_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let render_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("pchan_gpu::render_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        let vertex_buf = device.create_buffer(&BufferDescriptor {
            label: Some("pchan-gpu.vertex-buf"),
            size: mb(8) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vram_texture = device.create_texture(&TextureDescriptor {
            label: Some("pchan_gpu::vram_tex"),
            size: Extent3d {
                width: 1024,
                height: 512,
                ..Default::default()
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: VRAM_FMT,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[VRAM_FMT],
        });
        let vram_view = vram_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let vram_out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("psx.vram.out"),
            size: 1024 * 512 * size_of::<[u8; 4]>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let vram_blit_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("psx.vram.blit"),
            size: 1024 * 512 * size_of::<[u8; 4]>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("pchan_gpu::rasterizer_bind_group_layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: VRAM_FMT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
        let render_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pchan_gpu::rasterizer_bind_group"),
            layout: &render_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&vram_view),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&render_bind_group_layout)],
            immediate_size: 0,
        });
        let display_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("pchan_gpu::display_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let display_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("pchan_gpu::display::uniforms"),
            size: DisplayUniforms::DATASIZE as _,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let display_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pchan_gpu::display_bind_group"),
            layout: &display_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&render_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&render_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: display_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let display_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&display_bind_group_layout)],
            immediate_size: 0,
        });
        let render_pipelines = RenderPipelines {
            add: Self::create_render_pipeline(
                &device,
                &pipeline_layout,
                &shader_module,
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::OneMinusSrc1,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::REPLACE,
                }),
            ),
            subtract: Self::create_render_pipeline(
                &device,
                &pipeline_layout,
                &shader_module,
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::ReverseSubtract,
                    },
                    alpha: BlendComponent::REPLACE,
                }),
            ),
            subtract_opaque: Self::create_render_pipeline(
                &device,
                &pipeline_layout,
                &shader_module,
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Src1,
                        dst_factor: BlendFactor::OneMinusSrc1,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::REPLACE,
                }),
            ),
        };
        let display_shader = device.create_shader_module(include_wgsl!("../shaders/display.wgsl"));
        let display_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pchan_gpu::display_pipeline"),
            layout: Some(&display_pipeline_layout),
            vertex: VertexState {
                module: &display_shader,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &display_shader,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: display_format,
                    blend: None,
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let belt = StagingBelt::new(device.clone(), mb(4) as u64);
        let blit =
            create_blit_compute_resources(&device, &vram_blit_buffer, &render_view, &vram_view);
        let readback = create_readback_resources(&device, &render_view);
        Ok(Self {
            gpu: WgpuBackend {
                instance,
                adapter,
                device,
                queue,
                display_format,
            },
            res: BackendResources {
                render_pipelines,
                display_pipeline,
                vertex_buf,
                render_texture,
                render_view,
                render_bind_group,
                vram_texture,
                vram_out_buf,
                vram_blit_buffer,
                display_bind_group,
                display_uniform_buffer,
                belt: belt.into(),
                blit,
                readback,
            },
            conn: Conn {
                draw_call_chan: kanal::bounded_async(128),
                vram_out_chan: kanal::bounded_async(128),
            },
            tracy: TracyClient::default(),
            display_uniforms: DisplayUniforms::default(),
            clear_display_flag: AtomicBool::new(false),
        })
    }

    pub async fn try_new() -> Result<Self, InitError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            flags: InstanceFlags::from_env_or_default(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options: BackendOptions::from_env_or_default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: Features::default()
                    | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | Features::DUAL_SOURCE_BLENDING
                    | Features::CLEAR_TEXTURE
                    | Features::IMMEDIATES,
                required_limits: Limits {
                    max_immediate_size: 8,
                    ..default()
                },
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await?;
        Self::from_wgpu(WgpuBackend {
            instance,
            adapter,
            device,
            queue,
            display_format: TextureFormat::Bgra8UnormSrgb,
        })
    }

    pub async fn new() -> Self {
        Self::try_new().await.unwrap()
    }

    pub fn connect_emu<A: Allocator + Copy>(&self, emu: &mut Emu<A>) {
        emu.gpu.conn = self.conn.clone();
    }

    fn set_hw_display(&self, hw_dp: HWDisplayUniforms) {
        *self.display_uniforms.hw.lock().unwrap() = hw_dp;
    }

    pub fn start(self: Arc<Self>) {
        std::thread::Builder::new()
            .name("pchan-gpu-hw".into())
            .spawn(move || {
                self.tracy.set_thread_name("pchan-gpu");
                smol::block_on(self.render_loop());
            })
            .expect("failed to start hw rasterizer thread");
    }

    async fn render_loop(self: Arc<Self>) {
        tracing::info!("started gpu renderer task");
        let mut alloc = BlinkAlloc::new();
        loop {
            let msg = self
                .conn
                .draw_call_chan
                .1
                .recv()
                .await
                .expect("render channel closed");
            match msg {
                DrawCallMsg::DrawCall(draw_call) => {
                    alloc.reset();
                    let mut pass = Pass::new_in(self.res.belt.try_lock().unwrap(), &alloc);
                    pass.add_draw_call(draw_call);
                    match self.draw_submission_loop(&mut pass).await {
                        SubmissionResult::Reset(ack) => {
                            self.reset();
                            ack.send(()).await.unwrap();
                            continue;
                        }
                        SubmissionResult::RequestedVram(sink, ack) => {
                            self.readback(sink).await;
                            ack.send(()).await.unwrap();
                        }
                        SubmissionResult::Flush(flush) => {
                            self.update_hw_display_uniforms(&flush.display, flush.gpustat);
                        }
                    };

                    let start = Instant::now();
                    let mut render_pass = self.create_render_pass(pass);
                    render_pass.draw();
                    render_pass.finish();
                    let elapsed = start.elapsed();

                    // if tracing::enabled!(Level::DEBUG) {
                    tracing::info!("finished render in {:.2}ms", elapsed.as_millis_f32());
                    // }
                }
                DrawCallMsg::Flush(flush) => {
                    self.update_hw_display_uniforms(&flush.display, flush.gpustat);
                }
                DrawCallMsg::UpdateDisplay(display, gpustat) => {
                    self.update_hw_display_uniforms(&display, gpustat);
                }
                DrawCallMsg::Reset(ack) => {
                    self.reset();
                    ack.send(()).await.unwrap();
                }
                DrawCallMsg::RequestVram(sink, ack) => {
                    self.readback(sink).await;
                    ack.send(()).await.unwrap();
                }
            }
        }
    }

    fn update_hw_display_uniforms(&self, display: &Display, gpustat: GpuStatReg) {
        let mut uniforms = self.display_uniforms.hw.lock().unwrap();
        let new_uniforms = HWDisplayUniforms::from_hw(display, gpustat);

        if uniforms.dp_res != new_uniforms.dp_res {
            self.clear_display_flag
                .store(true, core::sync::atomic::Ordering::Release);
        }

        *uniforms = new_uniforms;
    }

    #[must_use]
    async fn draw_submission_loop<A: Allocator + Copy>(
        &self,
        pass: &mut Pass<'_, A>,
    ) -> SubmissionResult {
        loop {
            let msg = self
                .conn
                .draw_call_chan
                .1
                .recv()
                .await
                .expect("render channel closed");
            match msg {
                DrawCallMsg::DrawCall(draw_call) => {
                    pass.add_draw_call(draw_call);
                }
                DrawCallMsg::Flush(flush_msg) => {
                    pass.flush_geometry();
                    pass.flush_cpu_to_vram();
                    return SubmissionResult::Flush(flush_msg);
                }
                DrawCallMsg::UpdateDisplay(display, gpustat) => {
                    self.update_hw_display_uniforms(&display, gpustat);
                }
                DrawCallMsg::Reset(ack) => return SubmissionResult::Reset(ack),
                DrawCallMsg::RequestVram(sink, ack) => {
                    return SubmissionResult::RequestedVram(sink, ack);
                }
            }
        }
    }

    pub fn reset(&self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let ranges = ImageSubresourceRange {
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        };
        encoder.clear_texture(&self.res.render_texture, &ranges);
        encoder.clear_texture(&self.res.vram_texture, &ranges);
        encoder.clear_buffer(&self.res.vram_blit_buffer, 0, None);
        encoder.clear_buffer(&self.res.vram_out_buf, 0, None);
        let six = self.gpu.queue.submit([encoder.finish()]);
        self.gpu
            .device
            .poll(wgt::PollType::Wait {
                submission_index: Some(six),
                timeout: None,
            })
            .unwrap();
    }

    pub fn consume_clear_display(&self) -> bool {
        self.clear_display_flag.compare_exchange(
            true,
            false,
            core::sync::atomic::Ordering::Acquire,
            core::sync::atomic::Ordering::Acquire,
        ) == Ok(true)
    }

    pub fn upload_vram(&self, vram: &[u16]) {
        let vram =
            unsafe { slice::from_raw_parts(vram.as_ptr().cast::<u8>(), mem::size_of_val(vram)) };
        let mut belt = self.res.belt.lock().unwrap();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let vram_buf = belt.allocate(
            NonZeroU64::new(mb(2) as u64).unwrap(),
            NonZeroU64::new(256).unwrap(),
        );
        vram_buf
            .slice(0..vram.len() as u64)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(vram);

        let copy_size = Extent3d {
            width: 1024,
            height: 512,
            depth_or_array_layers: 1,
        };
        let source = TexelCopyBufferInfo {
            buffer: vram_buf.buffer(),
            layout: TexelCopyBufferLayout {
                offset: vram_buf.offset(),
                bytes_per_row: Some(1024u32 * size_of::<u16>() as u32),
                rows_per_image: Some(512),
            },
        };
        encoder.copy_buffer_to_texture(source, self.res.render_texture.as_image_copy(), copy_size);
        encoder.copy_buffer_to_texture(source, self.res.vram_texture.as_image_copy(), copy_size);

        belt.finish();
        let six = self.gpu.queue.submit([encoder.finish()]);
        belt.recall();
        self.gpu
            .device
            .poll(wgt::PollType::Wait {
                submission_index: Some(six),
                timeout: None,
            })
            .unwrap();
    }

    pub async fn readback(&self, sink: &VramSink) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        cpass.set_bind_group(0, &self.res.readback.bind_group, &[]);
        cpass.set_pipeline(&self.res.readback.pipe);
        cpass.dispatch_workgroups(1024 / 2 / 16, 512 / 16, 1);
        drop(cpass);

        encoder.copy_buffer_to_buffer(
            &self.res.readback.readback_buf,
            0,
            &self.res.vram_out_buf,
            0,
            Some(self.res.readback.readback_buf.size()),
        );
        self.gpu.queue.submit([encoder.finish()]);

        self.res
            .vram_out_buf
            .map_async(MapMode::Read, .., move |res| {
                if let Err(err) = res {
                    tracing::warn!("{err}");
                }
            });

        let device = self.gpu.device.clone();
        smol::unblock(move || {
            _ = device.poll(PollType::wait_indefinitely());
        })
        .await;
        let buf_mapped = self.res.vram_out_buf.get_mapped_range(..).unwrap();

        let (prefix, buf, _suffix) = unsafe { buf_mapped.align_to::<u16>() };
        assert_eq!(prefix, []);

        let mut vram = sink.lock().await;
        vram.copy_from_slice(buf);
        drop(buf_mapped);
        self.res.vram_out_buf.unmap();
    }
}

enum SubmissionResult {
    Reset(kanal::AsyncSender<()>),
    RequestedVram(&'static VramSink, kanal::AsyncSender<()>),
    Flush(FlushMsg),
}

#[repr(C)]
struct DisplayUniformData {
    dp_start: UVec2,
    dp_res: UVec2,
    screen_rect: UVec2,
    dp_debug: u32,
    dp_srgb: u32,
    dp_color_depth: u32,
    _pad: u32,
}

impl DisplayUniforms {
    pub const DATASIZE: usize = size_of::<DisplayUniformData>();
    fn to_data(&self) -> DisplayUniformData {
        let hw = self.hw.lock().unwrap();
        let app = self.app.lock().unwrap();
        let HWDisplayUniforms {
            dp_start,
            dp_res,
            dp_color_depth,
        } = *hw;
        let AppDisplayUniforms {
            screen_rect,
            dp_debug,
            dp_srgb,
        } = *app;
        DisplayUniformData {
            dp_start: dp_start.as_uvec2(),
            dp_res: dp_res.as_uvec2(),
            screen_rect: screen_rect.as_uvec2(),
            dp_debug: u32::from(dp_debug),
            dp_srgb: u32::from(dp_srgb),
            dp_color_depth: dp_color_depth.raw_value().as_u32(),
            _pad: 0u32,
        }
    }
}

#[derive(Debug)]
struct BlitV2VResources {
    uniforms_buf: Buffer,
    bind_group: BindGroup,
    pipe: ComputePipeline,
}

fn create_blit_compute_resources(
    device: &Device,
    blit_buffer: &Buffer,
    target_view: &TextureView,
    vram_view: &TextureView,
) -> BlitV2VResources {
    let shader = device.create_shader_module(include_wgsl!("../shaders/blit.wgsl"));

    let uniforms_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("pchan-gpu.blit.uniform_buffer"),
        size: CPU_TO_VRAM_UNIFORM_BUF_SIZE as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // uniforms
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // src buffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // dest render storage texture
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: VRAM_FMT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // dest vram storage texture
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: VRAM_FMT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: blit_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(target_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(vram_view),
            },
        ],
    });

    let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 8,
    });
    let cpipe = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: None,
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    BlitV2VResources {
        uniforms_buf: uniforms_buffer,
        bind_group,
        pipe: cpipe,
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct BlitUniforms {
    kind: BlitKind,
    gpustat: GpuStatReg,
    src: UVec2,
    dest: UVec2,
    size: UVec2,
}

const _: () = assert!(mem::size_of::<BlitUniforms>() == 32);
const _: () = assert!(mem::offset_of!(BlitUniforms, src) == 8);
const _: () = assert!(mem::offset_of!(BlitUniforms, size) == 24);

#[repr(u32)]
#[derive(Debug, Clone)]
enum BlitKind {
    CpuToVram = 0,
    VramToVram = 1,
}

impl BlitUniforms {
    fn from_vram_blit(value: &Gp0VramBlit, gpustat: GpuStatReg) -> Self {
        let mut size = value.size;
        if size.x == 0 {
            size.x = 1024;
        }
        if size.y == 0 {
            size.y = 512;
        }
        Self {
            kind: BlitKind::VramToVram,
            gpustat,
            src: uvec2(u32::from(value.src.x), u32::from(value.src.y)),
            dest: uvec2(u32::from(value.dest.x), u32::from(value.dest.y)),
            size: uvec2(u32::from(size.x), u32::from(size.y)),
        }
    }

    fn from_host_blit(value: Gp0HostBlit, gpustat: GpuStatReg) -> Self {
        Self {
            kind: BlitKind::CpuToVram,
            gpustat,
            src: UVec2::ZERO,
            dest: uvec2(u32::from(value.dest.x), u32::from(value.dest.y)),
            size: uvec2(u32::from(value.size.x), u32::from(value.size.y)),
        }
    }
}

impl BlitUniforms {
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(ptr::from_ref(self).cast::<u8>(), mem::size_of_val(self)) }
    }
}

#[derive(Debug)]
struct ReadbackResources {
    readback_buf: Buffer,
    bind_group: BindGroup,
    pipe: ComputePipeline,
}

fn create_readback_resources(device: &Device, target_view: &TextureView) -> ReadbackResources {
    let shader = include_wgsl!("../shaders/rgb8-to-rgb5.wgsl");
    let shader = device.create_shader_module(shader);

    let readback_buf = device.create_buffer(&BufferDescriptor {
        label: Some("pchan-gpu.blit.uniform_buffer"),
        size: mb(1) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // src texture
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // dest buffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(target_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(readback_buf.as_entire_buffer_binding()),
            },
        ],
    });

    let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let cpipe = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipe_layout),
        module: &shader,
        entry_point: None,
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    ReadbackResources {
        readback_buf,
        bind_group,
        pipe: cpipe,
    }
}
