pub(crate) mod render_pass;

use std::mem::{offset_of, transmute};

use arbitrary_int::prelude::*;
use color_eyre::eyre::bail;
use glam::{I16Vec2, U8Vec2, U8Vec3, U8Vec4, U16Vec2, UVec2, i16vec2, u8vec2, u16vec2};
use pchan_emu::gpu::draw_call::{
    DrawCall, DrawCallKind, DrawPolygon, DrawRect, DrawRectColor, RectSize, Shading, Uv,
};
use pchan_emu::gpu::{Conn, GpuStatReg, TextureColorMode, VramCoord, create_vram};
use pchan_emu::{Bus, Emu};
use wgpu::*;

#[derive(Debug, Clone)]
pub struct Renderer {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipeline_layout: PipelineLayout,
    render_pipeline: RenderPipeline,
    render_texture: Texture,
    render_view: TextureView,
    vram_texture: Texture,
    bind_group: BindGroup,

    conn: Conn<DrawCall>,
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
                | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::R16Uint],
        });
        let render_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                    attributes: &[
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
                    ],
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
        })
    }

    pub async fn new() -> Self {
        Self::try_new().await.unwrap()
    }

    pub fn connect_emu(&mut self, emu: &mut Emu) {
        emu.gpu_mut().conn = self.conn.clone();
    }

    pub fn start(self) {
        std::thread::spawn(move || {
            smol::block_on(async {
                tracing::info!("started gpu renderer task");
                loop {
                    tracing::trace!("waiting for draw calls...");
                    match self.conn.draw_call_chan.1.recv().await {
                        Ok(draw_calls) => {
                            tracing::info!(
                                "received {} draw_calls: {:#?}",
                                draw_calls.len(),
                                draw_calls
                            );
                            tracing::info!("waiting on vram...");
                            let Ok(mut vram) = self.conn.vram_in_chan.1.recv().await else {
                                continue;
                            };
                            tracing::info!("received vram");
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
}

#[derive(Debug, Clone, Default)]
pub struct Scene {
    vertex_buf: Vec<Vertex>,
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
    pub fn new_from_draw_calls(cmds: &[DrawCall]) -> Scene {
        let mut scene = Scene::default();
        for cmd in cmds {
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

pub async fn test_gpu(emu: &mut Emu) -> color_eyre::Result<()> {
    color_eyre::install()?;

    let renderer = Renderer::try_new().await?;
    let scene = Scene::new_from_draw_calls(&[
        // Solid color rect — top-left area
        DrawCall {
            gpustat: GpuStatReg::default(),
            inner: DrawCallKind::Rect(DrawRect {
                color: DrawRectColor::default().with_rgb(u24::new(0xff0000)),
                vertex1: VramCoord::new(0, 0),
                uv: None,
                var_size: Some(VramCoord { x: 64, y: 64 }),
            }),
        },
        // Tall narrow rect — stresses vertical spans
        DrawCall {
            gpustat: GpuStatReg::default(),
            inner: DrawCallKind::Rect(DrawRect {
                color: DrawRectColor::default().with_rgb(u24::new(0xFF00FF)),
                vertex1: VramCoord::new(300, 0),
                uv: None,
                var_size: Some(VramCoord { x: 8, y: 256 }),
            }),
        },
    ]);
    let mut pass = renderer.create_render_pass(scene).await;
    pass.draw(&emu.gpu.vram);
    pass.finish(&mut emu.gpu.vram).await?;

    let scene = Scene::new_from_draw_calls(&[
        // Solid color rect — different hue, overlapping slightly
        DrawCall {
            gpustat: GpuStatReg::default(),
            inner: DrawCallKind::Rect(DrawRect {
                color: DrawRectColor::default().with_rgb(u24::new(0x00ff00)),
                vertex1: VramCoord::new(48, 48),
                uv: None,
                var_size: Some(VramCoord { x: 64, y: 64 }),
            }),
        },
        // Wide short rect — stresses horizontal spans
        DrawCall {
            gpustat: GpuStatReg::default(),
            inner: DrawCallKind::Rect(DrawRect {
                color: DrawRectColor::default().with_rgb(u24::new(0x0000FF)),
                vertex1: VramCoord::new(0, 200),
                uv: None,
                var_size: Some(VramCoord { x: 512, y: 8 }),
            }),
        },
    ]);
    let mut pass = renderer.create_render_pass(scene).await;
    pass.draw(&emu.gpu.vram);
    pass.finish(&mut emu.gpu.vram).await?;

    Ok(())
}

#[cfg(test)]
mod test {
    use arbitrary_int::prelude::*;
    use bitbybit::bitfield;
    use pchan_emu::Emu;
    use pchan_utils::init_tracing;
    use ratatui::{
        crossterm,
        prelude::*,
        widgets::canvas::{Canvas, Rectangle},
    };
    use smol::Executor;

    use crate::test_gpu;

    #[test]
    fn run_gpu_test() -> color_eyre::Result<()> {
        init_tracing().call();
        let exec = Executor::new();
        let mut emu = Emu::default();
        smol::block_on(exec.run(test_gpu(&mut emu)))?;

        ratatui::run(|term| {
            loop {
                term.draw(|frame| {
                    let v = VramCanvasWidget {
                        vram: &emu.gpu.vram,
                        ..Default::default()
                    };

                    v.render(frame.area(), frame.buffer_mut());
                });

                if crossterm::event::read()?.is_key_press() {
                    break Ok(());
                }
            }
        })
    }

    #[derive(Debug, Default, Clone)]
    pub struct VramCanvasWidget<'a> {
        style: Style,
        vram: &'a [u16],
    }

    impl<'a> Widget for VramCanvasWidget<'a> {
        fn render(self, area: Rect, buf: &mut Buffer)
        where
            Self: Sized,
        {
            #[bitfield(u16)]
            struct Pixel {
                #[bits(0..=4, rw)]
                r: u5,
                #[bits(5..=9, rw)]
                g: u5,
                #[bits(10..=14, rw)]
                b: u5,
            }
            let canvas = Canvas::default()
                .x_bounds([0.0, 1024.0])
                .y_bounds([0.0, 512.0])
                .marker(symbols::Marker::Octant)
                .paint(|ctx| {
                    for y in 0..512 {
                        for x in 0..1024 {
                            let vram = &self.vram;
                            let vram_addr = 1024 * y + x;
                            let pixel = vram[vram_addr];
                            let pixel = Pixel::new_with_raw_value(pixel);

                            ctx.draw(&Rectangle {
                                x: x as f64,
                                y: y as f64,
                                width: 1.0,
                                height: 1.0,
                                color: Color::Rgb(
                                    (pixel.r().as_::<u16>() * 255 / 31) as u8,
                                    (pixel.g().as_::<u16>() * 255 / 31) as u8,
                                    (pixel.b().as_::<u16>() * 255 / 31) as u8,
                                ),
                            });
                        }
                    }
                });
            canvas.render(area, buf);
        }
    }
}
