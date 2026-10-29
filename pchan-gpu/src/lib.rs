#![feature(duration_millis_float)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(dead_code)]
#![allow(recursion_depth_exceeding_limit)]

extern crate alloc;

use alloc::slice;
use arbitrary_int::prelude::*;
use blink_alloc::BlinkAlloc;
pub use glam;
use pchan_emu::gpu::vram::{IVramCoord, VramCoord};
use pchan_emu::memory::mb;
use pchan_utils::default;
use pchan_utils::tracy::TracyClient;
pub(crate) mod render_pass;

use alloc::sync::Arc;
use core::alloc::Allocator;
use core::num::NonZeroU64;
use core::ops::Range;
use core::{mem, ptr};
use std::mem::offset_of;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;
use wgpu::util::StagingBelt;

use glam::prelude::*;
use pchan_emu::Emu;
use pchan_emu::gpu::draw_call::{
    DrawCall, DrawCallKind, DrawPolygon, DrawRect, Gp0HostBlit, Gp0QuickFill, Gp0VramBlit,
    GpuInternalDrawReg, RectSize, Shading,
};
use pchan_emu::gpu::{
    Conn, Display, DisplayColorDepth, DrawCallMsg, FlushMsg, Gp0TexWindowCmd, GpuStatReg,
    TextureColorMode, VramSink,
};
use thiserror::Error;
pub use wgpu;
use wgpu::*;

#[derive(Debug)]
pub struct Renderer {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub tracy: TracyClient,

    _pipeline_layout: PipelineLayout,
    render_pipelines: RenderPipelines,
    pub display_pipeline: RenderPipeline,
    _display_format: TextureFormat,
    pub vertex_buf: Buffer,
    pub render_texture: Texture,
    pub render_view: TextureView,
    render_bind_group: BindGroup,
    vram_texture: Texture,
    vram_out_buf: Buffer,
    vram_blit_buffer: Buffer,
    pub display_bind_group: BindGroup,
    pub display_uniform_buffer: Buffer,
    pub display_uniforms: DisplayUniforms,
    belt: Mutex<StagingBelt>,
    blit: BlitV2VResources,
    readback: ReadbackResources,

    conn: Conn,
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
        instance: Instance,
        adapter: Adapter,
        device: Device,
        queue: Queue,
        display_format: TextureFormat,
    ) -> Result<Self, InitError> {
        let shader_module = device.create_shader_module(include_wgsl!("../shaders/psx-gp0.wgsl"));

        // output render texture
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

        // uniform vram texture
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

        // uniforms

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

        // attributes

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
        let blit_v2v =
            create_blit_compute_resources(&device, &vram_blit_buffer, &render_view, &vram_view);
        let readback = create_readback_resources(&device, &render_view);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            _pipeline_layout: pipeline_layout,
            render_pipelines,
            render_texture,
            render_view,
            vram_texture,
            render_bind_group,
            conn: Conn {
                draw_call_chan: kanal::bounded_async(128),
                vram_out_chan: kanal::bounded_async(128),
            },
            display_pipeline,
            display_bind_group,
            display_uniform_buffer,
            display_uniforms: DisplayUniforms::default(),
            _display_format: display_format,
            tracy: TracyClient::default(),
            vram_out_buf,
            vertex_buf,
            belt: belt.into(),
            vram_blit_buffer,
            blit: blit_v2v,
            readback,
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
        Self::from_wgpu(
            instance,
            adapter,
            device,
            queue,
            TextureFormat::Bgra8UnormSrgb,
        )
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
                    let mut pass = Pass::new_in(self.belt.try_lock().unwrap(), &alloc);
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
        *self.display_uniforms.hw.lock().unwrap() = HWDisplayUniforms::from_hw(display, gpustat);
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
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let ranges = ImageSubresourceRange {
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        };
        encoder.clear_texture(&self.render_texture, &ranges);
        encoder.clear_texture(&self.vram_texture, &ranges);
        encoder.clear_buffer(&self.vram_blit_buffer, 0, None);
        encoder.clear_buffer(&self.vram_out_buf, 0, None);
        let six = self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgt::PollType::Wait {
                submission_index: Some(six),
                timeout: None,
            })
            .unwrap();
    }

    pub fn upload_vram(&self, vram: &[u16]) {
        let vram =
            unsafe { slice::from_raw_parts(vram.as_ptr().cast::<u8>(), mem::size_of_val(vram)) };
        let mut belt = self.belt.lock().unwrap();
        let mut encoder = self
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
        encoder.copy_buffer_to_texture(source, self.render_texture.as_image_copy(), copy_size);
        encoder.copy_buffer_to_texture(source, self.vram_texture.as_image_copy(), copy_size);

        belt.finish();
        let six = self.queue.submit([encoder.finish()]);
        belt.recall();
        self.device
            .poll(wgt::PollType::Wait {
                submission_index: Some(six),
                timeout: None,
            })
            .unwrap();
    }

    pub async fn readback(&self, sink: &VramSink) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        cpass.set_bind_group(0, &self.readback.bind_group, &[]);
        cpass.set_pipeline(&self.readback.pipe);
        cpass.dispatch_workgroups(1024 / 2 / 16, 512 / 16, 1);
        drop(cpass);

        encoder.copy_buffer_to_buffer(
            &self.readback.readback_buf,
            0,
            &self.vram_out_buf,
            0,
            Some(self.readback.readback_buf.size()),
        );
        self.queue.submit([encoder.finish()]);

        self.vram_out_buf.map_async(MapMode::Read, .., move |res| {
            if let Err(err) = res {
                tracing::warn!("{err}");
            }
        });

        let device = self.device.clone();
        smol::unblock(move || {
            _ = device.poll(PollType::wait_indefinitely());
        })
        .await;
        let buf_mapped = self.vram_out_buf.get_mapped_range(..).unwrap();

        let (prefix, buf, _suffix) = unsafe { buf_mapped.align_to::<u16>() };
        assert_eq!(prefix, []);

        let mut vram = sink.lock().await;
        vram.copy_from_slice(buf);

        {
            use std::io::Write;

            let width = 1024;
            let height = 512;
            let path = "/tmp/vram-dump.ppm";
            let mut out = Vec::with_capacity(width * height * 3 + 32);
            write!(out, "P6\n{width} {height}\n255\n").unwrap();

            // 5-bit -> 8-bit, replicating the top bits so 31 maps to 255
            let expand = |c: u16| ((c << 3) | (c >> 2)) as u8;

            for &px in vram.iter().take(width * height) {
                let r = px & 0x1F;
                let g = (px >> 5) & 0x1F;
                let b = (px >> 10) & 0x1F;
                out.extend_from_slice(&[expand(r), expand(g), expand(b)]);
            }

            if let Err(err) = std::fs::write(path, out) {
                tracing::warn!("failed to write {path}: {err}");
            }
        }

        drop(buf_mapped);
        self.vram_out_buf.unmap();
    }
}

enum SubmissionResult {
    Reset(kanal::AsyncSender<()>),
    RequestedVram(&'static VramSink, kanal::AsyncSender<()>),
    Flush(FlushMsg),
}

#[derive(derive_more::Debug, Default)]
enum Command<A: Allocator> {
    DrawGeometry(GeometryDraw<A>),
    VramToVramBlit {
        blit: Gp0VramBlit,
        gpustat: GpuStatReg,
    },
    CpuToVram(CpuToVramBlitBatch<A>),
    #[default]
    Nop,
}

struct GeometryDraw<A: Allocator> {
    batches: Vec<BatchedGeometryDraw, A>,
    scene: Scene<A>,
    dp: HWDisplayUniforms,
}

#[derive(derive_more::Debug)]
struct BatchedGeometryDraw {
    kind: BatchedGeometryDrawKind,
    span: Range<u32>,
}

#[derive(derive_more::Debug, PartialEq, Clone, Copy)]
enum BatchedGeometryDrawKind {
    OpaqueOnly,
    Transparent(Transparency),
}

#[derive(derive_more::Debug, PartialEq, Clone, Copy)]
enum Transparency {
    Add(f32),
    Average,
    Subtract,
}

struct CpuToVramBlit {
    blit: Gp0HostBlit,
    gpustat: GpuStatReg,
    cursor: u32,
    index: u32,
}

struct CpuToVramBlitBatchBuilder<A: Allocator> {
    cursor: u32,
    staging_buf: Buffer,
    staging_offset: BufferAddress,
    blits: Vec<CpuToVramBlit, A>,
    uniforms: Vec<BlitUniforms, A>,
}

struct CpuToVramBlitBatch<A: Allocator> {
    staging_buf: Buffer,
    staging_offset: BufferAddress,
    data_len: u64,
    blits: Vec<CpuToVramBlit, A>,
    uniforms: Vec<BlitUniforms, A>,
}

#[derive(derive_more::Debug)]
pub(crate) struct Pass<'b, A: Allocator> {
    cmds: Vec<Command<A>, A>,
    builder: PassCmdBuilder<A>,
    vertex_buf_cursor: u32,
    belt: MutexGuard<'b, StagingBelt>,
    #[debug(skip)]
    alloc: A,
}

enum PassCmdBuilder<A: Allocator> {
    None,
    Geometry(GeometryBuilder<A>),
    CpuToVramBlit(CpuToVramBlitBatchBuilder<A>),
}

#[derive(derive_more::Debug)]
struct GeometryBuilder<A: Allocator> {
    builder: BatchedGeometryDraw,
    batches: Vec<BatchedGeometryDraw, A>,
    scene: Scene<A>,
    aabbs: Vec<Aabb, A>,
    dp: HWDisplayUniforms,
}

impl<A: Allocator + Copy> Scene<A> {
    fn empty(at: u32, alloc: A) -> Self {
        Self {
            vertex_buf: Vec::with_capacity_in(512, alloc),
            vertex_buf_start: at,
            aabb: Aabb::void(),
        }
    }
    fn from_draw_call(at: u32, draw_call: DrawCall, alloc: A) -> Option<Self> {
        let mut scene = Scene {
            vertex_buf: Vec::new_in(alloc),
            vertex_buf_start: at,
            aabb: Aabb::void(),
        };
        match draw_call.inner {
            DrawCallKind::Rect(draw_rect) => {
                scene
                    .add_draw_rect_draw_call(
                        &draw_rect,
                        draw_call.gpustat,
                        &draw_call.draw_reg,
                        draw_call.tex_window,
                    )
                    .unwrap();
                Some(scene)
            }
            DrawCallKind::Polygon(draw_polygon) => {
                scene.add_draw_polygon_draw_call(
                    &draw_polygon,
                    draw_call.gpustat,
                    &draw_call.draw_reg,
                    draw_call.tex_window,
                );
                Some(scene)
            }
            DrawCallKind::Line(_draw_line) => {
                tracing::warn!("todo: DrawCall Line");
                Some(scene)
            }
            _ => None,
        }
    }
}

impl From<u2> for Transparency {
    fn from(value: u2) -> Self {
        match value.value() {
            0 => Transparency::Average,
            1 => Transparency::Add(1.0),
            2 => Transparency::Subtract,
            3 => Transparency::Add(0.25),
            _ => unreachable!("u2 cannot be greater than 3"),
        }
    }
}

fn draw_call_transparency(draw: &DrawCall) -> Option<Transparency> {
    if draw.inner.is_transparent() {
        let gpustat_mode = Transparency::from(draw.gpustat.semi_transparency());
        match &draw.inner {
            DrawCallKind::Polygon(poly) => Some(
                poly.semi_transparency()
                    .map_or(gpustat_mode, Transparency::from),
            ),
            _ => Some(gpustat_mode),
        }
    } else {
        None
    }
}

impl From<&DrawCall> for BatchedGeometryDrawKind {
    fn from(value: &DrawCall) -> Self {
        match draw_call_transparency(value) {
            Some(mode) => BatchedGeometryDrawKind::Transparent(mode),
            None => BatchedGeometryDrawKind::OpaqueOnly,
        }
    }
}

impl BatchedGeometryDraw {
    fn new(start: u32, kind: BatchedGeometryDrawKind) -> Self {
        BatchedGeometryDraw {
            kind,
            span: start..start,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Aabb {
    min: I16Vec2,
    max: I16Vec2,
}

impl Aabb {
    const VRAM: Self = Aabb {
        min: i16vec2(0, 0),
        max: i16vec2(1024, 512),
    };
    const fn void() -> Self {
        Self {
            min: I16Vec2::MAX,
            max: I16Vec2::MIN,
        }
    }
    const fn zero() -> Self {
        Self {
            min: I16Vec2::ZERO,
            max: I16Vec2::ZERO,
        }
    }
    fn new(origin: I16Vec2, size: I16Vec2) -> Self {
        debug_assert!(size.cmpge(I16Vec2::ZERO).all(), "negative size");
        let max = origin + size;
        Self { min: origin, max }
    }
    #[must_use]
    fn grow_to_include(self, pos: I16Vec2) -> Self {
        let min = self.min.min(pos);
        let max = self.max.max(pos);
        Self { min, max }
    }
    fn intersects(self, o: Aabb) -> bool {
        self.min.cmplt(o.max).all() && o.min.cmplt(self.max).all()
    }
    fn union(self, o: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }
    fn clamp(self, o: Aabb) -> Aabb {
        Aabb {
            min: self.min.clamp(o.min, o.max),
            max: self.max.clamp(o.min, o.max),
        }
    }
    fn bounding_box_of(draw_call: &DrawCall) -> Option<Self> {
        match &draw_call.inner {
            DrawCallKind::Rect(draw_rect) => match draw_rect.header.size() {
                RectSize::VarSize => Some(Aabb::new(
                    draw_rect.vertex1.as_i16vec2(),
                    draw_rect.var_size.map_or(I16Vec2::ZERO, |c| c.as_i16vec2()),
                )),
                RectSize::SinglePixel => {
                    Some(Aabb::new(draw_rect.vertex1.as_i16vec2(), i16vec2(1, 1)))
                }
                RectSize::Sprite8x8 => {
                    Some(Aabb::new(draw_rect.vertex1.as_i16vec2(), i16vec2(8, 8)))
                }
                RectSize::Sprite16x16 => {
                    Some(Aabb::new(draw_rect.vertex1.as_i16vec2(), i16vec2(16, 16)))
                }
            },
            DrawCallKind::Polygon(draw_polygon) => Some(
                draw_polygon
                    .attrs
                    .iter()
                    .fold(Aabb::void(), |aabb, attr| aabb.grow_to_include(attr.vertex)),
            ),
            DrawCallKind::Line(draw_line) => {
                let aabb = draw_line
                    .attrs
                    .iter()
                    .fold(Aabb::void(), |aabb, attr| aabb.grow_to_include(attr.vertex));
                Some(aabb)
            }
            _ => None,
        }
    }

    fn to_origin_size(self) -> (I16Vec2, I16Vec2) {
        (self.min, self.max - self.min)
    }

    fn to_vram_origin_size(self) -> (UVec2, UVec2) {
        let (origin, size) = self.to_origin_size();
        let cast = |x: i16| u32::from(x.max(0).cast_unsigned());

        let ox = cast(origin.x);
        let oy = cast(origin.y);
        let max_w = 1024u32.saturating_sub(ox);
        let max_h = 512u32.saturating_sub(oy);

        let width = cast(size.x).next_multiple_of(4).min(max_w);
        let height = cast(size.y).next_multiple_of(4).min(max_h);

        (uvec2(ox, oy), uvec2(width, height))
    }
}

impl<A: Allocator + Copy> GeometryBuilder<A> {
    fn new(start: u32, dp: HWDisplayUniforms, alloc: A) -> Self {
        Self {
            builder: BatchedGeometryDraw::new(start, BatchedGeometryDrawKind::OpaqueOnly),
            batches: Vec::with_capacity_in(16, alloc),
            scene: Scene::empty(start, alloc),
            aabbs: Vec::with_capacity_in(512, alloc),
            dp,
        }
    }
    fn add_draw_call(&mut self, draw_call: DrawCall) {
        match &mut self.builder.kind {
            BatchedGeometryDrawKind::OpaqueOnly => {
                if let Some(tp) = draw_call_transparency(&draw_call) {
                    self.begin_transparent(tp);
                    return self.add_draw_call(draw_call);
                }
            }
            BatchedGeometryDrawKind::Transparent(tp) => match draw_call_transparency(&draw_call) {
                Some(new_tp) if new_tp == *tp => {}
                Some(new_tp) => {
                    self.begin_transparent(new_tp);
                    return self.add_draw_call(draw_call);
                }
                None => {
                    self.begin_opaque();
                    return self.add_draw_call(draw_call);
                }
            },
        };
        match draw_call.inner {
            DrawCallKind::Rect(draw_rect) => {
                let len = self
                    .scene
                    .add_draw_rect_draw_call(
                        &draw_rect,
                        draw_call.gpustat,
                        &draw_call.draw_reg,
                        draw_call.tex_window,
                    )
                    .unwrap();
                let end = len as u32 + self.scene.vertex_buf_start;
                self.builder.span.end = end;
            }
            DrawCallKind::Polygon(draw_polygon) => {
                let len = self.scene.add_draw_polygon_draw_call(
                    &draw_polygon,
                    draw_call.gpustat,
                    &draw_call.draw_reg,
                    draw_call.tex_window,
                );
                let end = len as u32 + self.scene.vertex_buf_start;
                self.builder.span.end = end;
            }
            DrawCallKind::Line(_) => {
                tracing::warn!("todo: DrawCallKind::Line");
            }
            DrawCallKind::QuickFill(quick_fill) => {
                let len = self.scene.add_quick_fill_draw_call(
                    &quick_fill,
                    draw_call.gpustat,
                    &draw_call.draw_reg,
                    draw_call.tex_window,
                );
                let end = len as u32 + self.scene.vertex_buf_start;
                self.builder.span.end = end;
            }
            DrawCallKind::BlitVram2Vram(_)
            | DrawCallKind::BlitCpu2Vram(_, _)
            | DrawCallKind::BlitVram2Cpu(_)
            | DrawCallKind::Nop => {}
        }
    }

    fn begin_transparent(&mut self, tp: Transparency) {
        self.flush_batch();
        self.builder.kind = BatchedGeometryDrawKind::Transparent(tp);
    }

    fn begin_opaque(&mut self) {
        self.flush_batch();
        self.builder.kind = BatchedGeometryDrawKind::OpaqueOnly;
    }

    fn flush_batch(&mut self) {
        if self.builder.span.is_empty() {
            self.builder.kind = BatchedGeometryDrawKind::OpaqueOnly;
            self.aabbs.clear();
            return;
        }
        let end = self.builder.span.end;
        let draw = mem::replace(
            &mut self.builder,
            BatchedGeometryDraw::new(end, BatchedGeometryDrawKind::OpaqueOnly),
        );
        self.batches.push(draw);
        self.aabbs.clear();
    }
}

const CPU_TO_VRAM_BLIT_BUF_SIZE: usize = mb(2);
const CPU_TO_VRAM_UNIFORM_BUF_SIZE: usize = mb(2);
const CPU_TO_VRAM_MAX_BLITS: usize = CPU_TO_VRAM_UNIFORM_BUF_SIZE / size_of::<BlitUniforms>();

impl<A: Allocator + Copy> CpuToVramBlitBatchBuilder<A> {
    fn new(pass: &mut Pass<A>) -> Self {
        let buf = pass.belt.allocate(
            NonZeroU64::new(CPU_TO_VRAM_BLIT_BUF_SIZE as u64).unwrap(),
            NonZeroU64::new(4).unwrap(),
        );
        let offset = buf.offset();
        Self {
            cursor: 0,
            blits: Vec::with_capacity_in(64, pass.alloc),
            uniforms: Vec::with_capacity_in(64, pass.alloc),
            staging_buf: buf.buffer().clone(),
            staging_offset: offset,
        }
    }

    fn add_draw_call(
        &mut self,
        blit: Gp0HostBlit,
        data: Vec<u8>,
        gpustat: GpuStatReg,
    ) -> Result<(), Vec<u8>> {
        if self.uniforms.len() > CPU_TO_VRAM_MAX_BLITS {
            return Err(data);
        }

        let cursor = self.cursor;
        let end = cursor + data.len() as u32;

        self.cursor = end;

        let buf_start = u64::from(cursor) + self.staging_offset;
        let buf_end = u64::from(end) + self.staging_offset;
        // FIXME: fix buffer alignment issues
        self.staging_buf
            .get_mapped_range_mut(buf_start..buf_end)
            .unwrap()
            .copy_from_slice(&data);

        self.blits.push(CpuToVramBlit {
            blit,
            gpustat,
            cursor,
            index: self.blits.len() as u32,
        });
        self.uniforms
            .push(BlitUniforms::from_host_blit(blit, gpustat));

        Ok(())
    }

    fn flush(self) -> Option<CpuToVramBlitBatch<A>> {
        if self.uniforms.is_empty() || self.blits.is_empty() {
            return None;
        }
        Some(CpuToVramBlitBatch {
            staging_buf: self.staging_buf,
            staging_offset: self.staging_offset,
            blits: self.blits,
            data_len: u64::from(self.cursor),
            uniforms: self.uniforms,
        })
    }
}

impl<'b, A: Allocator + Copy> Pass<'b, A> {
    pub fn empty(belt: MutexGuard<'b, StagingBelt>, alloc: A) -> Self {
        let cmds = Vec::new_in(alloc);
        Pass {
            belt,
            cmds,
            builder: PassCmdBuilder::None,
            alloc,
            vertex_buf_cursor: 0,
        }
    }

    pub fn new_in(belt: MutexGuard<'b, StagingBelt>, alloc: A) -> Self {
        let cmds = Vec::new_in(alloc);
        Pass {
            cmds,
            builder: PassCmdBuilder::None,
            alloc,
            vertex_buf_cursor: 0,
            belt,
        }
    }

    pub fn add_draw_call(&mut self, mut draw_call: DrawCall) {
        match &mut self.builder {
            PassCmdBuilder::Geometry(builder) => match draw_call.inner {
                DrawCallKind::Nop => {}
                DrawCallKind::Rect(_)
                | DrawCallKind::Polygon(_)
                | DrawCallKind::Line(_)
                | DrawCallKind::QuickFill(_) => {
                    builder.add_draw_call(draw_call);
                }
                _ => {
                    self.flush_geometry();
                    self.add_draw_call(draw_call);
                }
            },
            PassCmdBuilder::CpuToVramBlit(builder) => match draw_call.inner {
                DrawCallKind::Nop => {}
                DrawCallKind::BlitCpu2Vram(gp0_host_blit, data) => {
                    match builder.add_draw_call(gp0_host_blit, data, draw_call.gpustat) {
                        Ok(_) => {}
                        Err(data) => {
                            tracing::warn!(
                                "ran out of space in cpu to vram blit staging buffer. maybe increase the capacity"
                            );
                            draw_call.inner = DrawCallKind::BlitCpu2Vram(gp0_host_blit, data);
                            self.flush_cpu_to_vram();
                            self.add_draw_call(draw_call);
                        }
                    }
                }
                _ => {
                    self.flush_cpu_to_vram();
                    self.add_draw_call(draw_call);
                }
            },
            PassCmdBuilder::None => match draw_call.inner {
                DrawCallKind::Nop => {}
                DrawCallKind::Rect(_)
                | DrawCallKind::Polygon(_)
                | DrawCallKind::Line(_)
                | DrawCallKind::QuickFill(_) => {
                    let dp = HWDisplayUniforms::from_hw(&draw_call.display, draw_call.gpustat);
                    let geometry_builder =
                        GeometryBuilder::new(self.vertex_buf_cursor, dp, self.alloc);
                    self.builder = PassCmdBuilder::Geometry(geometry_builder);
                    self.add_draw_call(draw_call);
                }
                DrawCallKind::BlitVram2Vram(blit) => {
                    self.cmds.push(Command::VramToVramBlit {
                        blit,
                        gpustat: draw_call.gpustat,
                    });
                }
                DrawCallKind::BlitCpu2Vram(_, _) => {
                    self.builder =
                        PassCmdBuilder::CpuToVramBlit(CpuToVramBlitBatchBuilder::new(self));
                    self.add_draw_call(draw_call)
                }
                DrawCallKind::BlitVram2Cpu(_blit) => todo!(),
            },
        }
    }

    fn flush_geometry(&mut self) {
        match self.builder.take() {
            PassCmdBuilder::Geometry(mut geom) => {
                geom.flush_batch();
                self.vertex_buf_cursor += geom.scene.vertex_bytes().len() as u32;
                self.cmds.push(Command::DrawGeometry(GeometryDraw {
                    batches: geom.batches,
                    scene: geom.scene,
                    dp: geom.dp,
                }));
            }
            other => {
                self.builder = other;
            }
        }
    }

    fn flush_cpu_to_vram(&mut self) {
        match self.builder.take() {
            PassCmdBuilder::CpuToVramBlit(builder) => {
                if let Some(cmd) = builder.flush() {
                    self.cmds.push(Command::CpuToVram(cmd));
                }
            }
            other => {
                self.builder = other;
            }
        }
    }
}

impl<A: Allocator> PassCmdBuilder<A> {
    fn take(&mut self) -> Self {
        mem::replace(self, PassCmdBuilder::None)
    }
}

// FIXME: pack this better
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Vertex {
    pos: I16Vec2,

    // these 2 must be packed together
    color: U8Vec3,
    color_mode: TextureColorMode,

    uv: U16Vec2,
    clut: U16Vec2,
    texpage_base: U8Vec2,
    flags: Flags,
    draw_area_top_left: VramCoord,
    draw_area_bottom_right: VramCoord,
    draw_offset: IVramCoord,
    gpustat: GpuStatReg,
    tex_window: Gp0TexWindowCmd,
}

struct VertexSpec {
    pos: I16Vec2,
    color: U8Vec3,
    color_mode: TextureColorMode,
    uv: U16Vec2,
    clut: U16Vec2,
    texpage_base: U8Vec2,
    flags: Flags,
    draw_area_top_left: VramCoord,
    draw_area_bottom_right: VramCoord,
    draw_offset: IVramCoord,
    gpustat: GpuStatReg,
    tex_window: Gp0TexWindowCmd,
}

impl Vertex {
    fn new(spec: VertexSpec) -> Self {
        Self {
            pos: spec.pos,
            color: spec.color,
            color_mode: spec.color_mode,
            uv: spec.uv,
            clut: spec.clut,
            texpage_base: spec.texpage_base,
            flags: spec.flags,
            draw_area_top_left: spec.draw_area_top_left,
            draw_area_bottom_right: spec.draw_area_bottom_right,
            draw_offset: spec.draw_offset,
            gpustat: spec.gpustat,
            tex_window: spec.tex_window,
        }
    }
    fn desc() -> &'static [VertexAttribute] {
        &[
            // position @location(0)
            VertexAttribute {
                format: VertexFormat::Sint16x2,
                offset: 0x0,
                shader_location: 0,
            },
            // color_and_mode @location(1)
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: offset_of!(Vertex, color) as _,
                shader_location: 1,
            },
            // clut @location(2)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, clut) as _,
                shader_location: 2,
            },
            // uv @location(3)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, uv) as _,
                shader_location: 3,
            },
            // texpage_base @location(4)
            VertexAttribute {
                format: VertexFormat::Uint8x2,
                offset: offset_of!(Vertex, texpage_base) as _,
                shader_location: 4,
            },
            // flags @location(5)
            VertexAttribute {
                format: VertexFormat::Uint8,
                offset: offset_of!(Vertex, flags) as _,
                shader_location: 5,
            },
            // draw_area_top_left @location(6)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, draw_area_top_left) as _,
                shader_location: 6,
            },
            // draw_area_bottom_right @location(7)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, draw_area_bottom_right) as _,
                shader_location: 7,
            },
            // draw_offset @location(8)
            VertexAttribute {
                format: VertexFormat::Sint16x2,
                offset: offset_of!(Vertex, draw_offset) as _,
                shader_location: 8,
            },
            // gpustat @location(9)
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: offset_of!(Vertex, gpustat) as _,
                shader_location: 9,
            },
            // tex_window @location(10)
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: offset_of!(Vertex, tex_window) as _,
                shader_location: 10,
            },
        ]
    }
}

#[bitbybit::bitfield(u8, debug)]
#[derive(Default)]
struct Flags {
    #[bit(0, rw)]
    dither: bool,
    #[bit(1, rw)]
    set_mask: bool,
    #[bit(2, rw)]
    quick_fill: bool,
    #[bit(3, rw)]
    textured: bool,
    #[bit(4, rw)]
    modulation: bool,
    #[bits(5..=6, rw)]
    semi_transparency: u2,
    #[bit(7, rw)]
    stp_on: bool,
}

impl Flags {
    pub const fn from_gpustat(gpustat: GpuStatReg) -> Self {
        Self::ZERO
            .with_set_mask(gpustat.set_mask())
            .with_dither(gpustat.dither())
            .with_semi_transparency(gpustat.semi_transparency())
    }

    pub const fn with_optional_semi_transparency(self, stp: Option<u2>) -> Self {
        match stp {
            Some(stp) => self.with_semi_transparency(stp),
            None => self,
        }
    }
}

#[derive(derive_more::Debug, Clone)]
pub struct Scene<A: Allocator> {
    vertex_buf: Vec<Vertex, A>,
    vertex_buf_start: u32,
    aabb: Aabb,
}

impl From<VramCoord> for Vertex {
    fn from(value: VramCoord) -> Self {
        Self {
            pos: i16vec2(value.x as i16, value.y as i16),
            ..Default::default()
        }
    }
}

impl Vertex {
    const SIZE: u64 = size_of::<Self>() as u64;

    pub fn with_pos(mut self, pos: I16Vec2) -> Self {
        self.pos = pos;
        self
    }
    pub fn with_color(mut self, color: U8Vec3) -> Self {
        self.color = color;
        self
    }
    pub fn with_uv(mut self, uv: U16Vec2) -> Self {
        self.uv = uv;
        self
    }
    pub fn with_clut(mut self, clut: U16Vec2) -> Self {
        self.clut = clut;
        self
    }
    pub fn repeat_tex_window(size: U16Vec2) -> U8Vec2 {
        size.as_u8vec2()
    }
    pub fn with_color_mode(mut self, mode: TextureColorMode) -> Self {
        self.color_mode = mode;
        self
    }
}

#[derive(Debug)]
pub struct Quad {
    top_left: Vertex,
    top_right: Vertex,
    bottom_left: Vertex,
    bottom_right: Vertex,
}

impl Quad {
    fn new(top_left: Vertex, size: U16Vec2) -> Self {
        let tex_size = size;
        Quad {
            top_left,
            top_right: top_left
                .with_pos(top_left.pos + i16vec2(size.x as i16, 0))
                .with_uv(top_left.uv.wrapping_add(u16vec2(tex_size.x, 0))),
            bottom_left: top_left
                .with_pos(top_left.pos + i16vec2(0, size.y as i16))
                .with_uv(top_left.uv.wrapping_add(u16vec2(0, tex_size.y))),
            bottom_right: top_left
                .with_pos(top_left.pos + i16vec2(size.x as i16, size.y as i16))
                .with_uv(top_left.uv.wrapping_add(u16vec2(tex_size.x, tex_size.y))),
        }
    }
    fn vertices(self) -> [Vertex; 4] {
        [
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        ]
    }
    fn triangulate(self) -> [Vertex; 6] {
        [
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_left,
            self.top_right,
            self.bottom_right,
        ]
    }
}

#[derive(Error, Debug)]
enum DrawRectError {
    #[error("draw rect: malformed call, missing draw size")]
    MissingVarSize,
}

impl<A: Allocator> Scene<A> {
    fn vertex_bytes(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self.vertex_buf.as_ptr().cast::<u8>(),
                mem::size_of_val(self.vertex_buf.as_slice()),
            )
        }
    }

    fn add_draw_rect_draw_call(
        &mut self,
        draw_rect: &DrawRect,
        gpustat: GpuStatReg,
        draw_reg: &GpuInternalDrawReg,
        tex_window: Gp0TexWindowCmd,
    ) -> Result<usize, DrawRectError> {
        let rgb = draw_rect.header.rgb().to_ne_bytes();
        let color_mode = match draw_rect.header.textured() {
            true => gpustat.texpage_colors(),
            false => TextureColorMode::C24BitDirect,
        };
        let color = U8Vec3::from_array(rgb);
        let uv = draw_rect.uv.unwrap_or_default();
        let flags = Flags::from_gpustat(gpustat)
            .with_textured(draw_rect.header.textured())
            .with_modulation(!draw_rect.header.raw())
            .with_stp_on(draw_rect.header.semi_transparent());
        let top_left = Vertex::new(VertexSpec {
            pos: i16vec2(draw_rect.vertex1.x as i16, draw_rect.vertex1.y as i16),
            color,
            color_mode,
            uv: uv.uv.as_u16vec2(),
            clut: uv.extra_as_clut(),
            texpage_base: gpustat.texpage_base(),
            flags,
            draw_area_top_left: draw_reg.draw_area_top_left,
            draw_area_bottom_right: draw_reg.draw_area_bottom_right,
            draw_offset: draw_reg.draw_offset,
            gpustat,
            tex_window,
        });

        let quad: Quad = match (draw_rect.header.size(), draw_rect.var_size) {
            (RectSize::VarSize, None) => return Err(DrawRectError::MissingVarSize),
            (RectSize::VarSize, Some(size)) => Quad::new(top_left, u16vec2(size.x, size.y)),
            (RectSize::SinglePixel, _) => Quad::new(top_left, u16vec2(1, 1)),
            (RectSize::Sprite8x8, _) => Quad::new(top_left, u16vec2(8, 8)),
            (RectSize::Sprite16x16, _) => Quad::new(top_left, u16vec2(16, 16)),
        };
        let vertices = triangulate_quad(&quad.vertices());
        for vertex in &vertices {
            self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
        }
        self.vertex_buf.extend(vertices);
        Ok(self.vertex_bytes().len())
    }

    fn add_draw_polygon_draw_call(
        &mut self,
        draw_polygon: &DrawPolygon,
        gpustat: GpuStatReg,
        draw_reg: &GpuInternalDrawReg,
        tex_window: Gp0TexWindowCmd,
    ) -> usize {
        let header = draw_polygon.header;
        let clut = draw_polygon.clut;
        let texpage = draw_polygon.texpage;
        let texpage_base = texpage.unwrap_or_default().to_u8vec2();

        let color = header.color();
        let shading = header.shading();
        let color_mode = match (header.textured(), texpage) {
            (true, Some(texpage)) => texpage.texpage_colors(),
            _ => TextureColorMode::C15BitDirect,
        };
        let flags = Flags::from_gpustat(gpustat)
            .with_optional_semi_transparency(draw_polygon.semi_transparency())
            .with_modulation(header.modulation())
            .with_stp_on(header.semi_transparent())
            .with_dither(gpustat.dither() && (header.modulation() || header.goraud()))
            .with_textured(header.textured());
        let mut vertices = draw_polygon
            .attrs
            .iter()
            .map(|attr| {
                let mut spec = VertexSpec {
                    pos: attr.vertex,
                    color: U8Vec3::from_array(color.to_ne_bytes()),
                    color_mode,
                    uv: attr.uv.unwrap_or_default().uv.as_u16vec2(),
                    texpage_base,
                    clut,
                    flags,
                    draw_area_top_left: draw_reg.draw_area_top_left,
                    draw_area_bottom_right: draw_reg.draw_area_bottom_right,
                    draw_offset: draw_reg.draw_offset,
                    gpustat,
                    tex_window,
                };
                match shading {
                    Shading::Flat => {}
                    Shading::Gouraud => {
                        spec.color = attr
                            .color
                            .unwrap_or(U8Vec3::from_array(color.to_ne_bytes()));
                    }
                }
                Vertex::new(spec)
            })
            .collect::<heapless::Vec<_, 4>>();
        match header.vertex_count() {
            pchan_emu::gpu::draw_call::DrawPolygonVertexCount::Three => {
                assert_eq!(vertices.len(), 3);
                ensure_vertex_order(&mut vertices, [0, 1, 2]);
                for vertex in &vertices {
                    self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
                }
                self.vertex_buf.extend(vertices);
            }
            pchan_emu::gpu::draw_call::DrawPolygonVertexCount::Four => {
                assert_eq!(vertices.len(), 4);
                let vertices = triangulate_quad(&vertices);

                for vertex in &vertices {
                    self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
                }

                self.vertex_buf.extend_from_slice(&vertices);
            }
        };
        self.vertex_bytes().len()
    }

    fn add_quick_fill_draw_call(
        &mut self,
        fill: &Gp0QuickFill,
        gpustat: GpuStatReg,
        _: &GpuInternalDrawReg,
        tex_window: Gp0TexWindowCmd,
    ) -> usize {
        let flags = Flags::from_gpustat(gpustat)
            .with_modulation(false)
            .with_stp_on(false)
            .with_dither(false)
            .with_textured(false)
            .with_quick_fill(true);

        let topleft = Vertex::new(VertexSpec {
            pos: fill.pos.fill_cmd_pos_mask().as_i16vec2(),
            color: fill.color,
            color_mode: TextureColorMode::C15BitDirect,
            uv: default(),
            clut: default(),
            texpage_base: default(),
            flags,
            draw_area_top_left: default(),
            draw_area_bottom_right: VramCoord::new(1024, 512),
            draw_offset: default(),
            gpustat,
            tex_window,
        });

        let quad = Quad::new(topleft, fill.size.fill_cmd_size_mask().as_u16vec2());
        let mut vertices = triangulate_quad(&quad.vertices());

        ensure_vertex_order(&mut vertices, [0, 1, 2]);
        for vertex in &vertices {
            self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
        }
        self.vertex_buf.extend(vertices);

        self.vertex_bytes().len()
    }
}

/// PSX vertex order according to PSX SPX by Martin Korth
///
/// <https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#notes>
fn triangulate_quad(vertices: &[Vertex]) -> [Vertex; 6] {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];
    [v0, v1, v2, v1, v2, v3]
}

/// Credits to jsgroth at <https://jsgroth.dev/blog/posts/ps1-diamond/#preparing-the-scene>
fn ensure_vertex_order(vertex_buf: &mut [Vertex], indices: [usize; 3]) {
    let [v0, v1, v2] = vertex_buf
        .get_disjoint_mut(indices)
        .expect("indices must be disjoint");
    let v0_pos = v0.pos.as_i64vec2();
    let v1_pos = v1.pos.as_i64vec2();
    let v2_pos = v2.pos.as_i64vec2();
    let cross_product_z = (v1_pos.x - v0_pos.x) * (v2_pos.y - v0_pos.y)
        - (v1_pos.y - v0_pos.y) * (v2_pos.x - v0_pos.x);
    if cross_product_z < 0 {
        mem::swap(v0, v1);
    }
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
