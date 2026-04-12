pub(crate) mod render_pass;

use std::mem::offset_of;
use std::sync::{Arc, Mutex};

use color_eyre::eyre::bail;
use glam::{I16Vec2, U8Vec2, U8Vec3, U16Vec2, UVec2, i16vec2, u8vec2, u16vec2};
use pchan_emu::gpu::draw_call::{
    DrawCallCollection, DrawCallKind, DrawPolygon, DrawRect, RectSize, Shading,
};
use pchan_emu::gpu::{Conn, GpuStatReg, TextureColorMode, VramCoord};
use pchan_emu::{Bus, Emu};
use pchan_utils::Chan;
use wgpu::*;

#[derive(Debug)]
pub struct Renderer {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipeline_layout: PipelineLayout,
    render_pipeline: RenderPipeline,
    pub display_pipeline: RenderPipeline,
    pub render_texture: Texture,
    pub render_view: TextureView,
    vram_texture: Texture,
    bind_group: BindGroup,
    pub display_bind_group: BindGroup,
    pub display_uniform_buffer: Buffer,
    pub display_uniforms: Mutex<DisplayUniforms>,

    conn: Conn,
}

#[derive(Debug, Clone)]
pub enum UpdateUniforms {
    Display(DisplayUniforms),
}

#[derive(Debug, Clone, Default)]
pub struct DisplayUniforms {
    pub dp_start: U16Vec2,
    pub dp_res: U16Vec2,
    pub screen_rect: U16Vec2,
    pub dp_debug: bool,
}

impl Renderer {
    pub async fn try_new() -> color_eyre::Result<Self> {
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
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: Features::default(),
                required_limits: Limits::defaults(),
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await?;
        let shader_module =
            device.create_shader_module(include_wgsl!("../shaders/draw_call_rect.wgsl"));

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
            format: TextureFormat::R16Uint,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::R16Uint],
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

        // uniform vram texture
        let vram_texture = device.create_texture(&TextureDescriptor {
            label: Some("pchan_gpu::render_tex"),
            size: Extent3d {
                width: 512,
                height: 512,
                ..Default::default()
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::R32Uint],
        });
        let vram_view = vram_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // uniforms

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pchan_gpu::rasterizer_bind_group_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: TextureFormat::R32Uint,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pchan_gpu::rasterizer_bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&vram_view),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
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
                            sample_type: TextureSampleType::Uint,
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
            label: Some("pchan_gpu::rasterizer_bind_group"),
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

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64, // 8
                    step_mode: VertexStepMode::Vertex,
                    attributes: Vertex::desc(),
                }],
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
                module: &shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::R16Uint,
                    blend: None,
                    write_mask: ColorWrites::default(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let display_shader = device.create_shader_module(include_wgsl!("../shaders/display.wgsl"));
        let display_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pchan_gpu::display_pipeline"),
            layout: Some(&display_pipeline_layout),
            vertex: VertexState {
                module: &display_shader,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: 0,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[],
                }],
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
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            pipeline_layout,
            render_pipeline,
            render_texture,
            render_view,
            vram_texture,
            bind_group,
            conn: Conn {
                draw_call_chan: kanal::bounded_async(0),
                vram_in_chan: kanal::bounded_async(0),
                vram_out_chan: kanal::bounded_async(1),
            },
            display_pipeline,
            display_bind_group,
            display_uniform_buffer,
            display_uniforms: Mutex::new(DisplayUniforms::default()),
        })
    }

    pub async fn new() -> Self {
        Self::try_new().await.unwrap()
    }

    pub fn connect_emu(&mut self, emu: &mut Emu) {
        emu.gpu_mut().conn = self.conn.clone();
    }

    pub fn start(self: Arc<Self>) {
        std::thread::spawn(move || {
            smol::block_on(async {
                tracing::info!("started gpu renderer task");
                loop {
                    tracing::trace!("waiting for draw calls...");
                    match self.conn.draw_call_chan.1.recv().await {
                        Ok(draw_calls) => {
                            tracing::trace!(
                                "received {} draw_calls: {:#?}",
                                draw_calls.draw_calls.len(),
                                draw_calls
                            );
                            tracing::trace!("waiting on vram...");
                            let Ok(mut vram) = self.conn.vram_in_chan.1.recv().await else {
                                continue;
                            };
                            tracing::debug!("received vram");

                            let scene = Scene::new_from_draw_calls(&draw_calls);
                            let mut pass = self.create_render_pass(scene).await;
                            pass.draw(&vram);
                            if pass.finish(&mut vram).await.is_ok() {
                                _ = self.conn.vram_out_chan.0.send(vram).await;
                            }
                            tracing::info!("finished render");
                        }
                        Err(err) => {
                            tracing::error!(%err);
                            break;
                        }
                    }
                }
            });
        });
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

    uv: U8Vec2,
    _pad_02: [u8; 2],
    clut: U16Vec2,
    textured: bool,
    _pad: u8,
    texpage_base: U8Vec2,
    flags: Flags,
    _pad_03: [u8; 2],
}

impl Vertex {
    fn desc() -> &'static [VertexAttribute] {
        &[
            // position @location(0)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
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
                format: VertexFormat::Uint8x2,
                offset: offset_of!(Vertex, uv) as _,
                shader_location: 3,
            },
            // texpage_base @location(4)
            VertexAttribute {
                format: VertexFormat::Uint8x2,
                offset: offset_of!(Vertex, texpage_base) as _,
                shader_location: 4,
            },
            // textured @location(5)
            VertexAttribute {
                format: VertexFormat::Uint8,
                offset: offset_of!(Vertex, textured) as _,
                shader_location: 5,
            },
            // flags @location(6)
            VertexAttribute {
                format: VertexFormat::Uint8,
                offset: offset_of!(Vertex, flags) as _,
                shader_location: 6,
            },
        ]
    }
}

#[bitbybit::bitfield(u8, debug)]
#[derive(Default)]
struct Flags {
    #[bit(0, rw)]
    dither: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Scene {
    vertex_buf: Vec<Vertex>,
    dp_start: U16Vec2,
    dp_res: U16Vec2,
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
    pub fn with_uv(mut self, uv: U8Vec2) -> Self {
        self.uv = uv;
        self
    }
    pub fn with_clut(mut self, clut: U16Vec2) -> Self {
        self.clut = clut;
        self
    }
    pub fn repeat_tex_window(size: U16Vec2) -> U8Vec2 {
        (size % u16vec2(255, 255)).as_u8vec2()
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
    fn new_with_topleft_and_size(top_left: Vertex, size: U16Vec2) -> Self {
        let tex_size = Vertex::repeat_tex_window(size);
        Quad {
            top_left,
            top_right: top_left
                .with_pos(top_left.pos + i16vec2(size.x as i16, 0))
                .with_uv(top_left.uv + u8vec2(tex_size.x, 0)),
            bottom_left: top_left
                .with_pos(top_left.pos + i16vec2(0, size.y as i16))
                .with_uv(top_left.uv + u8vec2(0, tex_size.y)),
            bottom_right: top_left
                .with_pos(top_left.pos + i16vec2(size.x as i16, size.y as i16))
                .with_uv(top_left.uv + u8vec2(tex_size.x, tex_size.y)),
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

impl Scene {
    pub fn new_from_draw_calls(cmds: &DrawCallCollection) -> Scene {
        let Some(gpustat) = cmds.draw_calls.last().map(|draw| draw.gpustat) else {
            return Scene::default();
        };
        let mut scene = Scene {
            dp_res: gpustat.resolution(),
            dp_start: cmds.display.display_vram_start,
            ..Default::default()
        };
        for cmd in &cmds.draw_calls {
            match &cmd.inner {
                DrawCallKind::Rect(draw_rect) => {
                    // _ = scene.add_draw_rect_draw_call(draw_rect);
                }
                DrawCallKind::Polygon(draw_polygon) => {
                    _ = scene.add_draw_polygon_draw_call(draw_polygon, cmd.gpustat);
                }
                DrawCallKind::Line(draw_line) => {}
            }
        }

        scene
    }

    #[pchan_macros::instrument(skip_all, err)]
    fn add_draw_rect_draw_call(&mut self, draw_rect: &DrawRect) -> color_eyre::Result<()> {
        let top_left: Vertex = draw_rect.vertex1.into();
        let rgb = draw_rect.color.rgb().to_ne_bytes();
        let color_mode = match draw_rect.color.textured() {
            // TODO: pick the correct color mode for textured polygons
            true => TextureColorMode::C15BitDirect,
            false => TextureColorMode::C24BitDirect,
        };
        let top_left = top_left
            .with_color(U8Vec3::from_array(rgb))
            .with_color_mode(color_mode);

        let quad: Quad = match (draw_rect.color.size(), draw_rect.var_size) {
            (RectSize::VarSize, None) => bail!("malformed draw call: missing var size"),
            (RectSize::VarSize, Some(size)) => {
                Quad::new_with_topleft_and_size(top_left, u16vec2(size.x, size.y))
            }
            (RectSize::SinglePixel, _) => Quad::new_with_topleft_and_size(top_left, u16vec2(1, 1)),
            (RectSize::Sprite8x8, _) => Quad::new_with_topleft_and_size(top_left, u16vec2(8, 8)),
            (RectSize::Sprite16x16, _) => {
                Quad::new_with_topleft_and_size(top_left, u16vec2(16, 16))
            }
        };
        let mut vertices = triangulate_quad(&quad.vertices());
        ensure_vertex_order(&mut vertices, [0, 1, 2]);
        ensure_vertex_order(&mut vertices, [3, 4, 5]);
        self.vertex_buf.extend(vertices);
        Ok(())
    }

    #[pchan_macros::instrument(skip_all, err)]
    fn add_draw_polygon_draw_call(
        &mut self,
        draw_polygon: &DrawPolygon,
        gpustat: GpuStatReg,
    ) -> color_eyre::Result<()> {
        let header = draw_polygon.header;
        let clut = draw_polygon.clut;
        let texpage = draw_polygon.texpage;
        let texpage_base = texpage.to_u8vec2();
        // let texpage_base = u8vec2(
        //     gpustat.texpage_x_base().as_u8(),
        //     gpustat.texpage_y_base().as_u8(),
        // );

        let color = header.color();
        let shading = header.shading();
        let color_mode = match header.textured() {
            true => gpustat.texpage_colors(),
            false => TextureColorMode::C15BitDirect,
        };
        let flags = Flags::default()
            .with_dither(gpustat.dither() && (header.modulation() || header.goraud()));
        let mut vertices = match shading {
            // DONE: Goraud shading
            Shading::Flat => draw_polygon
                .attrs
                .iter()
                .map(|attr| Vertex {
                    pos: attr.vertex,
                    color: U8Vec3::from_array(color.to_ne_bytes()),
                    color_mode,
                    uv: attr.uv.unwrap_or_default().uv,
                    texpage_base,
                    textured: header.textured(),
                    _pad: 0,
                    clut,
                    _pad_02: [0; 2],
                    flags,
                    _pad_03: [0; 2],
                })
                .collect::<heapless::Vec<_, 4>>(),
            Shading::Gouraud => draw_polygon
                .attrs
                .iter()
                .map(|attr| Vertex {
                    pos: attr.vertex,
                    color: attr
                        .color
                        .unwrap_or(U8Vec3::from_array(color.to_ne_bytes())),
                    color_mode,
                    uv: attr.uv.unwrap_or_default().uv,
                    clut,
                    texpage_base,
                    textured: header.textured(),
                    _pad: 0,
                    _pad_02: [0; 2],
                    flags,
                    _pad_03: [0; 2],
                })
                .collect(),
        };
        match header.vertex_count() {
            pchan_emu::gpu::draw_call::DrawPolygonVertexCount::Three => {
                assert_eq!(vertices.len(), 3);
                ensure_vertex_order(&mut vertices, [0, 1, 2]);
                self.vertex_buf.extend(vertices);
            }
            pchan_emu::gpu::draw_call::DrawPolygonVertexCount::Four => {
                assert_eq!(vertices.len(), 4);
                let mut vertices = triangulate_quad(&vertices);
                // ensure_vertex_order(&mut vertices, [0, 1, 2]);
                // ensure_vertex_order(&mut vertices, [3, 4, 5]);

                self.vertex_buf.extend_from_slice(&vertices);
            }
        };

        Ok(())
    }
}

/// PSX vertex order according to PSX SPX by Martin Korth
///
/// https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#notes
fn triangulate_quad(vertices: &[Vertex]) -> [Vertex; 6] {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];
    [v0, v1, v2, v1, v2, v3]
}

/// Credits to jsgroth at https://jsgroth.dev/blog/posts/ps1-diamond/#preparing-the-scene
fn ensure_vertex_order(vertex_buf: &mut [Vertex], indices: [usize; 3]) {
    let [v0, v1, v2] = vertex_buf
        .get_disjoint_mut(indices)
        .expect("indices must be disjoint");
    let cross_product_z = (v1.pos.x - v0.pos.x) * (v2.pos.y - v0.pos.y)
        - (v1.pos.y - v0.pos.y) * (v2.pos.x - v0.pos.x);
    if cross_product_z < 0 {
        std::mem::swap(v0, v1);
    }
}

pub type DisplayUniformData = (UVec2, UVec2, UVec2, u64);

impl DisplayUniforms {
    pub const DATASIZE: usize = size_of::<DisplayUniformData>();
    fn to_data(&self) -> DisplayUniformData {
        let DisplayUniforms {
            dp_start,
            dp_res,
            screen_rect,
            dp_debug,
        } = self;
        (
            dp_start.as_uvec2(),
            dp_res.as_uvec2(),
            screen_rect.as_uvec2(),
            *dp_debug as u64,
        )
    }
}
