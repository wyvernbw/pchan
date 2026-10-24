use core::alloc::Allocator;
use core::num::NonZeroU64;
use core::{mem, ptr, slice};
use std::sync::MutexGuard;

use crate::{BatchedGeometryDraw, BlitUniforms, Command, Pass, Renderer, Scene, Vertex};
use pchan_emu::gpu::GpuStatReg;
use pchan_emu::gpu::draw_call::{Gp0HostBlit, Gp0VramBlit};
use pchan_utils::tracy;
use tracing::Level;
use wgpu::util::{BufferInitDescriptor, DeviceExt, StagingBelt};
use wgpu::*;

const WORKGROUP_SIZE: u32 = 16;

impl Renderer {
    pub(crate) fn create_render_pass<A: Allocator>(&self, pass: Pass<A>) -> RenderPass<'_, A> {
        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let belt = self.belt.try_lock().unwrap();
        RenderPass {
            encoder,
            renderer: self,
            pass,
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

#[derive(derive_more::Debug)]
pub struct RenderPass<'a, A: Allocator> {
    encoder: CommandEncoder,
    belt: MutexGuard<'a, StagingBelt>,
    renderer: &'a Renderer,
    pass: Pass<A>,
}

impl<A: Allocator + Copy> RenderPass<'_, A> {
    pub(crate) fn draw(&mut self, vram: &[u16], dirty: bool) {
        let _draw = tracy::span!("rd-gpu-draw");
        self.init_vram(vram, dirty);

        let pass = self.pass.take();
        for cmd in &pass.cmds {
            match cmd {
                Command::DrawGeometry(geom) => {
                    tracing::debug!("- geometry:");
                    self.start_scene(&geom.scene);
                    for batch in &geom.batches {
                        tracing::debug!("\t- batch: {}..{}", batch.span.start, batch.span.end);
                        self.draw_batch(batch, &geom.scene);
                    }
                }
                Command::VramToVramBlit { blit, gpustat } => {
                    self.vram_blit(blit, *gpustat);
                }
                Command::CpuToVramBlit {
                    blit,
                    data,
                    gpustat,
                } => {
                    self.cpu_to_vram_blit(*blit, *gpustat, data);
                }
            }
        }
    }

    fn init_vram(&mut self, vram: &[u16], dirty: bool) {
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
                    width: 1024,
                    height: 512,
                    depth_or_array_layers: 1,
                },
            );

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
        }
    }

    fn start_scene(&mut self, scene: &Scene<A>) {
        let vertex_start = scene.vertex_buf_start;
        let vertex_buf = scene.vertex_bytes();
        let vertex_end = vertex_start + vertex_buf.len() as u32;
        debug_assert!(vertex_start < vertex_end);

        if let Some(size) = NonZeroU64::new(vertex_buf.len() as u64) {
            let mut view = self.belt.write_buffer(
                &mut self.encoder,
                &self.renderer.vertex_buf,
                vertex_start.into(),
                size,
            );
            view.copy_from_slice(vertex_buf);
        }
    }

    fn draw_batch(&mut self, batch: &BatchedGeometryDraw, scene: &Scene<A>) {
        let vertex_count = scene.vertex_buf.len();
        let byte_start = batch.span.start;
        let byte_end = batch.span.end;
        let vertex_start = byte_start / size_of::<Vertex>() as u32;
        let vertex_end = byte_end / size_of::<Vertex>() as u32;

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
        tracing::debug!(
            "\tvertex range: {}..{} ({})",
            byte_start,
            byte_end,
            vertex_count
        );
        render_pass.set_pipeline(&self.renderer.render_pipeline);
        render_pass.set_bind_group(0, &self.renderer.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.renderer.vertex_buf.slice(..));
        render_pass.draw(vertex_start..vertex_end, 0..1);

        drop(render_pass);

        self.encoder.copy_texture_to_texture(
            self.renderer.render_texture.as_image_copy(),
            self.renderer.vram_texture.as_image_copy(),
            Extent3d {
                width: 1024,
                height: 512,
                depth_or_array_layers: 1,
            },
        );
    }

    fn vram_blit(&mut self, blit: &Gp0VramBlit, gpustat: GpuStatReg) {
        // copy render target to buffer because we can't have
        // concurrent reads and writes to the same texture storage
        self.encoder.copy_texture_to_buffer(
            self.renderer.render_texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &self.renderer.vram_blit_buffer,
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

        let uniforms = BlitUniforms::from_vram_blit(blit, gpustat);
        let uniforms_staging = self
            .belt
            .allocate(NonZeroU64::new(512).unwrap(), NonZeroU64::new(256).unwrap());
        uniforms_staging
            .get_mapped_range_mut()
            .unwrap()
            .slice(0..uniforms.as_bytes().len())
            .copy_from_slice(uniforms.as_bytes());

        self.encoder.copy_buffer_to_buffer(
            uniforms_staging.buffer(),
            uniforms_staging.offset(),
            &self.renderer.blit.uniforms_buf,
            0,
            uniforms.as_bytes().len().next_multiple_of(4) as u64,
        );

        let mut cpass = self.encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("psx.gpu.blit<vram->vram>"),
            timestamp_writes: None,
        });

        if tracing::enabled!(Level::TRACE) {
            tracing::trace!(?uniforms);
        }

        let dispatch_x = uniforms.size.x.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = uniforms.size.y.div_ceil(WORKGROUP_SIZE);

        cpass.set_pipeline(&self.renderer.blit.pipe);
        cpass.set_bind_group(0, &self.renderer.blit.bind_group, &[]);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(cpass)
    }

    fn cpu_to_vram_blit(&mut self, blit: Gp0HostBlit, gpustat: GpuStatReg, data: &[u8]) {
        let staging = self.belt.allocate(
            NonZeroU64::new(data.len().next_multiple_of(256) as u64).unwrap(),
            NonZeroU64::new(256).unwrap(),
        );
        staging
            .get_mapped_range_mut()
            .unwrap()
            .slice(0..data.len())
            .copy_from_slice(data);
        self.encoder.copy_buffer_to_buffer(
            staging.buffer(),
            staging.offset(),
            &self.renderer.vram_blit_buffer,
            0,
            data.len() as u64,
        );

        let uniforms = BlitUniforms::from_host_blit(blit, gpustat);
        let uniforms_staging = self
            .belt
            .allocate(NonZeroU64::new(512).unwrap(), NonZeroU64::new(256).unwrap());
        uniforms_staging
            .get_mapped_range_mut()
            .unwrap()
            .slice(0..uniforms.as_bytes().len())
            .copy_from_slice(uniforms.as_bytes());
        self.encoder.copy_buffer_to_buffer(
            uniforms_staging.buffer(),
            uniforms_staging.offset(),
            &self.renderer.blit.uniforms_buf,
            0,
            uniforms.as_bytes().len() as u64,
        );

        let mut cpass = self.encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("psx.gpu.blit<cpu->vram>"),
            timestamp_writes: None,
        });

        let dispatch_x = uniforms.size.x.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = uniforms.size.y.div_ceil(WORKGROUP_SIZE);

        if tracing::enabled!(Level::TRACE) {
            tracing::trace!(?uniforms);
        }
        cpass.set_pipeline(&self.renderer.blit.pipe);
        cpass.set_bind_group(0, &self.renderer.blit.bind_group, &[]);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(cpass);
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
                if let Err(err) = res {
                    tracing::warn!("{err}");
                }
            });
        let device = self.renderer.device.clone();
        smol::unblock(move || {
            _ = device.poll(PollType::wait_indefinitely());
        })
        .await;
        let buf_mapped = self.renderer.vram_out_buf.get_mapped_range(..).unwrap();
        let buf =
            unsafe { core::slice::from_raw_parts(buf_mapped.as_ptr().cast::<u16>(), 1024 * 512) };
        vram.copy_from_slice(buf);
        drop(buf_mapped);
        self.renderer.vram_out_buf.unmap();

        Ok(())
    }
}
