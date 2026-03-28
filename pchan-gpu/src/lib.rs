#![feature(iter_array_chunks)]
#![feature(proc_macro_hygiene)]
#![feature(try_blocks)]

use pchan_emu::Emu;
use pchan_emu::gpu::VramCoord;
use pchan_emu::memory::mb;
use smol::Executor;
use wgpu::BackendOptions;
use wgpu::Backends;
use wgpu::BindGroupLayoutDescriptor;
use wgpu::BindGroupLayoutEntry;
use wgpu::BindingType;
use wgpu::BufferUsages;
use wgpu::ColorTargetState;
use wgpu::ColorWrites;
use wgpu::CommandEncoderDescriptor;
use wgpu::DeviceDescriptor;
use wgpu::ExperimentalFeatures;
use wgpu::Extent3d;
use wgpu::Face;
use wgpu::Features;
use wgpu::FragmentState;
use wgpu::FrontFace;
use wgpu::Instance;
use wgpu::InstanceDescriptor;
use wgpu::InstanceFlags;
use wgpu::Limits;
use wgpu::LoadOp;
use wgpu::MapMode;
use wgpu::MemoryBudgetThresholds;
use wgpu::MemoryHints;
use wgpu::MultisampleState;
use wgpu::Operations;
use wgpu::Origin3d;
use wgpu::PipelineCompilationOptions;
use wgpu::PipelineLayoutDescriptor;
use wgpu::PolygonMode;
use wgpu::PowerPreference;
use wgpu::PrimitiveState;
use wgpu::PrimitiveTopology;
use wgpu::RenderPassColorAttachment;
use wgpu::RenderPipelineDescriptor;
use wgpu::RequestAdapterOptions;
use wgpu::ShaderStages;
use wgpu::StoreOp;
use wgpu::TexelCopyBufferInfo;
use wgpu::TexelCopyBufferLayout;
use wgpu::TexelCopyTextureInfoBase;
use wgpu::TextureAspect;
use wgpu::TextureDescriptor;
use wgpu::TextureDimension;
use wgpu::TextureFormat;
use wgpu::TextureUsages;
use wgpu::TextureView;
use wgpu::Trace;
use wgpu::VertexAttribute;
use wgpu::VertexBufferLayout;
use wgpu::VertexFormat;
use wgpu::VertexState;
use wgpu::VertexStepMode;
use wgpu::include_wgsl;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::wgt::BufferDescriptor;
use wgpu::wgt::PollType;

pub async fn test_gpu(emu: &mut Emu) -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: InstanceFlags::from_env_or_default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: BackendOptions::from_env_or_default(),
        display: None,
    });

    let mut adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;

    let (mut device, mut queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: Features::default(),
            required_limits: Limits::defaults(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await?;

    let features = adapter.get_texture_format_features(TextureFormat::R16Uint);
    tracing::info!("R16Uint usages: {:?}", features.allowed_usages);

    let render_target = device.create_texture(&TextureDescriptor {
        label: Some("render_tex"),
        size: Extent3d {
            width: 1024,
            height: 512,
            ..Default::default()
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R16Uint,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[TextureFormat::R16Uint],
    });
    let render_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let shader_module =
        device.create_shader_module(include_wgsl!("../shaders/draw_call_rect.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });

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
                        offset: 0,
                        shader_location: 0,
                    },
                    // color @location(1)
                    VertexAttribute {
                        format: VertexFormat::Uint16,
                        offset: 4,
                        shader_location: 1,
                    },
                ],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
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

    #[repr(C)]
    struct Vertex {
        pos: [u16; 2],
        color: u16,
        _pad: u16,
    }

    let quad_vertices = [
        Vertex {
            pos: [10, 10],
            color: 0x0f30,
            _pad: 0,
        },
        Vertex {
            pos: [512, 10],
            color: 0x001f,
            _pad: 0,
        },
        Vertex {
            pos: [10, 256],
            color: 0x0000,
            _pad: 0,
        },
        Vertex {
            pos: [10, 256],
            color: 0x7FFF,
            _pad: 0,
        },
        Vertex {
            pos: [512, 10],
            color: 0x00aa0,
            _pad: 0,
        },
        Vertex {
            pos: [512, 256],
            color: 0x7FFF,
            _pad: 0,
        },
    ];

    let quad_vertices = unsafe {
        let ptr = quad_vertices.as_ptr() as *const u8;
        std::slice::from_raw_parts(ptr, std::mem::size_of_val(&quad_vertices))
    };

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        usage: BufferUsages::VERTEX,
        contents: quad_vertices,
    });
    tracing::info!(
        "vertex buffer size: {} bytes (expected {})",
        vertex_buffer.size(),
        size_of::<Vertex>() * 6
    );

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &render_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color {
                        r: 0.5,
                        g: 0.25,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&render_pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
    }

    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("output"),
        size: mb(1) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfoBase {
            texture: &render_target,
            mip_level: 0,
            origin: Origin3d::default(),
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1024 * 2),
                rows_per_image: Some(512),
            },
        },
        Extent3d {
            width: 1024,
            height: 512,
            depth_or_array_layers: 1,
        },
    );

    queue.submit([encoder.finish()]);
    output_buffer.map_async(MapMode::Read, .., move |res| {
        res.unwrap();
    });
    device.poll(PollType::wait_indefinitely())?;
    let buf = &output_buffer.get_mapped_range(..)[..];

    assert!(!buf.iter().all(|value| *value == 0), "empty buffer");

    for y in 0..512usize {
        for x in 0..1024usize {
            let offset = (y * 1024 + x) * 2;
            let vram_addr = y * 1024 + x;
            let pixel = u16::from_ne_bytes([buf[offset], buf[offset + 1]]);
            emu.gpu.vram[vram_addr] = pixel;
        }
    }

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
                .marker(symbols::Marker::Quadrant)
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
