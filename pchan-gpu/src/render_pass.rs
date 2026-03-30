use crate::{Renderer, Scene, Vertex};
use pchan_emu::memory::mb;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

impl Renderer {
    pub async fn create_render_pass(&self, scene: Scene) -> RenderPass<'_> {
        let vertex_buf = unsafe {
            std::slice::from_raw_parts(
                scene.vertex_buf.as_slice() as *const [_] as *const u8,
                std::mem::size_of_val(scene.vertex_buf.as_slice()),
            )
        };
        let vertex_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::VERTEX,
            contents: vertex_buf,
        });
        tracing::info!(
            "vertex buffer size: {} bytes (expected {})",
            vertex_buffer.size(),
            size_of::<Vertex>() * 6
        );

        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        RenderPass {
            scene,
            encoder,
            renderer: self,
            vertex_buf: vertex_buffer,
        }
    }
}
#[derive(Debug)]
pub struct RenderPass<'a> {
    encoder: CommandEncoder,
    renderer: &'a Renderer,
    scene: Scene,
    vertex_buf: Buffer,
}

impl<'a> RenderPass<'a> {
    pub fn draw(&mut self) {
        if self.scene.vertex_buf.is_empty() {
            return;
        }

        let mut render_pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.renderer.render_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(&self.renderer.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        render_pass.draw(0..self.scene.vertex_buf.len() as u32, 0..1);
    }

    #[pchan_macros::instrument(err)]
    pub async fn finish(mut self, vram: &mut [u16]) -> color_eyre::Result<()> {
        let output_buffer = self.renderer.device.create_buffer(&BufferDescriptor {
            label: Some("output"),
            size: mb(1) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.encoder.copy_texture_to_buffer(
            TexelCopyTextureInfoBase {
                texture: &self.renderer.render_texture,
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

        self.renderer.queue.submit([self.encoder.finish()]);
        output_buffer.map_async(MapMode::Read, .., move |res| {
            res.unwrap();
        });
        let device = self.renderer.device.clone();
        smol::unblock(move || {
            _ = device.poll(PollType::wait_indefinitely());
        })
        .await;
        let buf = &output_buffer.get_mapped_range(..)[..];

        for y in 0..512usize {
            for x in 0..1024usize {
                let offset = (y * 1024 + x) * 2;
                let vram_addr = y * 1024 + x;
                let pixel = u16::from_ne_bytes([buf[offset], buf[offset + 1]]);
                vram[vram_addr] = pixel;
            }
        }

        Ok(())
    }
}
