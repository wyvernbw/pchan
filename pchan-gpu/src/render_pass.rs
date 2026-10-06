use core::num::NonZeroU64;
use core::{mem, ptr, slice};
use std::sync::MutexGuard;

use crate::{Renderer, Scene};
use pchan_emu::memory::mb;
use pchan_utils::tracy;
use wgpu::util::{BufferInitDescriptor, DeviceExt, StagingBelt};
use wgpu::*;

impl Renderer {
    pub fn create_render_pass(&self, scene: Scene) -> RenderPass<'_> {
        let vertex_buf = unsafe {
            slice::from_raw_parts(
                scene.vertex_buf.as_ptr().cast::<u8>(),
                mem::size_of_val(scene.vertex_buf.as_slice()),
            )
        };

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let mut belt = self.belt.try_lock().unwrap();
        if let Some(size) = NonZeroU64::new(vertex_buf.len() as u64) {
            let mut view = belt.write_buffer(&mut encoder, &self.vertex_buf, 0, size);
            view.copy_from_slice(vertex_buf);
        }

        RenderPass {
            scene,
            encoder,
            renderer: self,
            vertex_buf_len: vertex_buf.len() as u64,
            belt,
        }
    }

    pub fn draw_display(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let vertex_buf = &[0, 0, 0, 0, 0, 0];
        let vertex_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::VERTEX,
            contents: vertex_buf,
        });

        {
            let display_uniforms = self.display_uniforms.to_data();
            let display_uniforms_slice = unsafe {
                let len = size_of_val(&display_uniforms);
                let ptr = ptr::from_ref(&display_uniforms).cast::<u8>();
                slice::from_raw_parts(ptr, len)
            };
            self.queue
                .write_buffer(&self.display_uniform_buffer, 0, display_uniforms_slice);
        }

        render_pass.set_pipeline(&self.display_pipeline);
        render_pass.set_bind_group(0, &self.display_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
    }
}
#[derive(Debug)]
pub struct RenderPass<'a> {
    encoder: CommandEncoder,
    belt: MutexGuard<'a, StagingBelt>,
    renderer: &'a Renderer,
    scene: Scene,
    vertex_buf_len: u64,
}

impl RenderPass<'_> {
    pub fn draw(&mut self, vram: &[u16], dirty: bool) {
        let _draw = tracy::span!("rd-gpu-draw");
        if self.scene.vertex_buf.is_empty() {
            return;
        }
        let vram_buf =
            unsafe { slice::from_raw_parts(vram.as_ptr().cast::<u8>(), mem::size_of_val(vram)) };

        if dirty {
            const STAGING_SIZE: NonZeroU64 = NonZeroU64::new(1024 * 2 * 512).unwrap();
            const STAGING_ALIGNMENT: NonZeroU64 = NonZeroU64::new(256).unwrap();

            let staging = self.belt.allocate(STAGING_SIZE, STAGING_ALIGNMENT);
            let mut view = staging
                .get_mapped_range_mut()
                .expect("failed to map staging buffer");
            view.copy_from_slice(vram_buf);
            self.encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: staging.buffer(),
                    layout: TexelCopyBufferLayout {
                        offset: staging.offset(),
                        bytes_per_row: Some(1024 * 2),
                        rows_per_image: Some(512),
                    },
                },
                self.renderer.vram_texture.as_image_copy(),
                Extent3d {
                    width: 512,
                    height: 512,
                    depth_or_array_layers: 1,
                },
            );
            // self.renderer.queue.write_texture(
            //     self.renderer.vram_texture.as_image_copy(),
            //     vram_buf,
            //     TexelCopyBufferLayout {
            //         offset: 0,
            //         bytes_per_row: Some(1024 * 2),
            //         rows_per_image: Some(512),
            //     },
            //     Extent3d {
            //         width: 512,
            //         height: 512,
            //         depth_or_array_layers: 1,
            //     },
            // );

            self.encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: staging.buffer(),
                    layout: TexelCopyBufferLayout {
                        offset: staging.offset(),
                        bytes_per_row: Some(1024 * 2),
                        rows_per_image: Some(512),
                    },
                },
                self.renderer.render_texture.as_image_copy(),
                Extent3d {
                    width: 1024,
                    height: 512,
                    depth_or_array_layers: 1,
                },
            );

            // self.renderer.queue.write_texture(
            //     self.renderer.render_texture.as_image_copy(),
            //     vram_buf,
            //     TexelCopyBufferLayout {
            //         offset: 0,
            //         bytes_per_row: Some(1024 * 2),
            //         rows_per_image: Some(512),
            //     },
            //     Extent3d {
            //         width: 1024,
            //         height: 512,
            //         depth_or_array_layers: 1,
            //     },
            // );
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
        render_pass.set_bind_group(0, &self.renderer.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.renderer.vertex_buf.slice(..self.vertex_buf_len));
        render_pass.draw(0..self.scene.vertex_buf.len() as u32, 0..1);
    }

    pub async fn finish(mut self, vram: &mut [u16]) -> Result<(), MapRangeError> {
        let _commit = tracy::span!("rd-commit");
        self.encoder.copy_texture_to_buffer(
            TexelCopyTextureInfoBase {
                texture: &self.renderer.render_texture,
                mip_level: 0,
                origin: Origin3d::default(),
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &self.renderer.vram_out_buf,
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

        self.belt.finish();
        self.renderer.queue.submit([self.encoder.finish()]);
        self.belt.recall();
        self.renderer
            .vram_out_buf
            .map_async(MapMode::Read, .., move |res| {
                res.unwrap();
            });
        let device = self.renderer.device.clone();
        smol::unblock(move || {
            _ = device.poll(PollType::wait_indefinitely());
        })
        .await;
        let buf_mapped = self.renderer.vram_out_buf.get_mapped_range(..)?;
        let buf =
            unsafe { core::slice::from_raw_parts(buf_mapped.as_ptr().cast::<u16>(), 1024 * 512) };
        vram.copy_from_slice(buf);
        drop(buf_mapped);
        self.renderer.vram_out_buf.unmap();

        Ok(())
    }
}
