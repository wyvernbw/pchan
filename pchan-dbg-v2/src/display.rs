use pchan_gpu::Renderer;
use wgpu::TextureViewDescriptor;
use wgpu::*;

use crate::AppState;

pub struct DisplayState {
    pub output_tex: Texture,
    pub display_buf: Buffer,
}

impl DisplayState {
    pub fn new(gpu: &Renderer) -> Self {
        let size = gpu.display_uniforms.lock().unwrap().dp_res;
        let output_tex = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: size.x as u32,
                height: size.y as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Bgra8UnormSrgb],
        });
        let display_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size.x as u64 * size.y as u64 * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        DisplayState {
            output_tex,
            display_buf,
        }
    }

    pub fn configure(&mut self, state: &AppState) {
        let size = state.gpu.display_uniforms.lock().unwrap().screen_rect;
        let out_size = self.output_tex.size();
        if out_size.width != size.x as u32 || out_size.height != size.y as u32 {
            self.output_tex.destroy();
            *self = DisplayState::new(&state.gpu);
        }
    }
}

pub fn draw_display(state: &AppState, dp: &mut DisplayState) -> Vec<u8> {
    dp.configure(state);

    let gpu = &state.gpu;
    let window_surface_view = dp.output_tex.create_view(&TextureViewDescriptor {
        label: None,
        format: None,
        dimension: None,
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT),
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &window_surface_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        label: Some("display render pass"),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    gpu.draw_display(&mut rpass);
    drop(rpass);

    let unpadded = dp.output_tex.width() * 4;
    let padded = (unpadded + 255) & !255; // round up to 256
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfoBase {
            texture: &dp.output_tex,
            mip_level: 0,
            origin: Origin3d::default(),
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &dp.display_buf,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(dp.output_tex.height()),
            },
        },
        Extent3d {
            width: dp.output_tex.width(),
            height: dp.output_tex.height(),
            depth_or_array_layers: 1,
        },
    );

    gpu.queue.submit([encoder.finish()]);

    dp.display_buf.map_async(MapMode::Read, .., move |res| {
        res.unwrap();
    });
    _ = gpu.device.poll(PollType::wait_indefinitely());
    let render = {
        let buf = dp.display_buf.get_mapped_range(..);
        // println!("buf.len = {:?}", buf.len());
        // println!("buf all zeroes: {}", buf.iter().all(|b| *b == 0));
        // println!(
        //     "display uniforms = {:#?}",
        //     state.gpu.display_uniforms.lock().unwrap()
        // );
        let mut buf_owned = buf.to_vec();
        bgra_to_rgba(&mut buf_owned);
        {
            // let mut file = std::fs::File::create("./display.bmp").unwrap();
            // image::codecs::bmp::BmpEncoder::new(&mut file)
            //     .encode(
            //         &buf_owned,
            //         dp.output_tex.width(),
            //         dp.output_tex.height(),
            //         image::ExtendedColorType::Rgba8,
            //     )
            //     .unwrap();
        }
        buf_owned
    };
    dp.display_buf.unmap();
    render
}

fn bgra_to_rgba(data: &mut [u8]) {
    for pixel in data.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
}
