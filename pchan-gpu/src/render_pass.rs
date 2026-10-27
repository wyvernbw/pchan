use core::alloc::Allocator;
use core::num::NonZeroU64;
use core::{mem, ptr, slice};

use crate::{
    Aabb, BatchedGeometryDraw, BatchedGeometryDrawKind, BlitUniforms, Command, CpuToVramBlit,
    CpuToVramBlitBatch, Pass, Renderer, Scene, Transparency, Vertex,
};
use pchan_emu::gpu::GpuStatReg;
use pchan_emu::gpu::draw_call::Gp0VramBlit;
use pchan_emu::memory::kb;
use pchan_utils::tracy;
use tracing::Level;
use wgpu::*;

const WORKGROUP_SIZE: u32 = 16;

impl Renderer {
    pub(crate) fn create_render_pass<'b, A: Allocator>(
        &self,
        pass: Pass<'b, A>,
    ) -> RenderPass<'_, 'b, A> {
        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        RenderPass {
            encoder,
            renderer: self,
            pass,
        }
    }

    pub fn draw_display(&self, render_pass: &mut wgpu::RenderPass<'_>) {
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
        render_pass.draw(0..6, 0..1);
    }
}

#[derive(derive_more::Debug)]
pub struct RenderPass<'a, 'b, A: Allocator> {
    encoder: CommandEncoder,
    renderer: &'a Renderer,
    pass: Pass<'b, A>,
}

impl<A: Allocator + Copy> RenderPass<'_, '_, A> {
    pub(crate) fn draw(&mut self) {
        let _draw = tracy::span!("rd-gpu-draw");

        let cmds = mem::replace(&mut self.pass.cmds, Vec::new_in(self.pass.alloc));
        for cmd in cmds {
            match cmd {
                Command::DrawGeometry(geom) => {
                    tracing::debug!("- geometry ({} batches):", geom.batches.len());
                    self.start_scene(&geom.scene);
                    let mut render = self.start_render();
                    for batch in &geom.batches {
                        tracing::debug!("\t- batch: {}..{}", batch.span.start, batch.span.end);
                        render.draw_batch(batch);
                    }
                    drop(render);
                    self.end_scene(&geom.scene);
                }
                Command::VramToVramBlit { blit, gpustat } => {
                    self.vram_blit(&blit, gpustat);
                }
                Command::CpuToVram(batch) => {
                    tracing::debug!("- cpu->vram blits ({}):", batch.blits.len());
                    let mut pass = self.cpu_to_vram_blit_start(&batch);
                    for blit in &batch.blits {
                        tracing::debug!("\t- #{}: {}..", blit.index, blit.cursor);
                        pass.cpu_to_vram_blit(blit);
                    }
                    pass.cpu_to_vram_blit_end();
                }
                Command::Nop => {}
            }
        }
    }

    fn start_scene(&mut self, scene: &Scene<A>) {
        let vertex_start = scene.vertex_buf_start;
        let vertex_buf = scene.vertex_bytes();
        let vertex_end = vertex_start + vertex_buf.len() as u32;
        debug_assert!(vertex_start < vertex_end);

        if let Some(size) = NonZeroU64::new(vertex_buf.len() as u64) {
            let mut view = self.pass.belt.write_buffer(
                &mut self.encoder,
                &self.renderer.vertex_buf,
                vertex_start.into(),
                size,
            );
            view.copy_from_slice(vertex_buf);
        }
    }

    fn start_render(&mut self) -> GeometryRenderPass<'_> {
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
        render_pass.set_vertex_buffer(0, self.renderer.vertex_buf.slice(..));
        render_pass.set_pipeline(&self.renderer.render_pipelines.add);
        render_pass.set_bind_group(0, &self.renderer.render_bind_group, &[]);

        GeometryRenderPass {
            rpass: render_pass,
            current_kind: BatchedGeometryDrawKind::OpaqueOnly,
            gpu: self.renderer,
        }
    }

    fn copy_render_to_vram(&mut self, aabb: Aabb) {
        let (origin, size) = aabb.to_vram_origin_size();
        let copy_info = |texture| TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d {
                x: origin.x,
                y: origin.y,
                z: 0,
            },
            aspect: TextureAspect::All,
        };
        self.encoder.copy_texture_to_texture(
            copy_info(&self.renderer.render_texture),
            copy_info(&self.renderer.vram_texture),
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
    }

    fn end_scene(&mut self, scene: &Scene<A>) {
        self.copy_render_to_vram(scene.aabb);
    }

    fn commit_render_to_vram(&mut self) {
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

    fn commit_vram_to_render(&mut self) {
        self.encoder.copy_texture_to_texture(
            self.renderer.vram_texture.as_image_copy(),
            self.renderer.render_texture.as_image_copy(),
            Extent3d {
                width: 1024,
                height: 512,
                depth_or_array_layers: 1,
            },
        );
    }

    fn vram_blit(&mut self, blit: &Gp0VramBlit, gpustat: GpuStatReg) {
        let src_aabb = Aabb::new(blit.src.as_i16vec2(), blit.size.as_i16vec2());
        let (src_origin, src_size) = src_aabb.to_vram_origin_size();

        // copy render target to buffer because we can't have
        // concurrent reads and writes to the same texture storage
        self.encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &self.renderer.render_texture,
                origin: Origin3d {
                    x: src_origin.x,
                    y: src_origin.y,
                    z: 0,
                },
                mip_level: 0,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &self.renderer.vram_blit_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1024 * 4),
                    rows_per_image: Some(512),
                },
            },
            Extent3d {
                width: src_size.x,
                height: src_size.y,
                depth_or_array_layers: 1,
            },
        );

        let uniforms = BlitUniforms::from_vram_blit(blit, gpustat);
        let uniforms_staging = self
            .pass
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
        cpass.set_immediates(0, &0u32.to_le_bytes());
        cpass.set_immediates(4, &0u32.to_le_bytes());
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(cpass);

        self.commit_render_to_vram();
    }

    fn cpu_to_vram_blit_start<'a>(
        &'a mut self,
        batch: &'a CpuToVramBlitBatch<A>,
    ) -> CpuToVramPass<'a, A> {
        self.encoder.copy_buffer_to_buffer(
            &batch.staging_buf,
            batch.staging_offset,
            &self.renderer.vram_blit_buffer,
            0,
            batch.data_len,
        );

        let (prefix, uniforms, _) = unsafe { batch.uniforms.align_to::<u8>() };
        debug_assert!(prefix.is_empty(), "alignment cant be smaller than 1 byte.");

        let uniforms_staging = self.pass.belt.allocate(
            NonZeroU64::new(uniforms.len().next_multiple_of(4) as u64).unwrap(),
            NonZeroU64::new(4).unwrap(),
        );
        uniforms_staging
            .get_mapped_range_mut()
            .unwrap()
            .slice(0..uniforms.len())
            .copy_from_slice(uniforms);
        self.encoder.copy_buffer_to_buffer(
            uniforms_staging.buffer(),
            uniforms_staging.offset(),
            &self.renderer.blit.uniforms_buf,
            0,
            uniforms.len() as u64,
        );
        let mut cpass = self.encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("psx.gpu.blit<cpu->vram>"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.renderer.blit.pipe);
        cpass.set_bind_group(0, &self.renderer.blit.bind_group, &[]);

        CpuToVramPass { batch, cpass }
    }

    pub fn finish(mut self) {
        let _commit = tracy::span!("rd-commit");
        self.pass.belt.finish();
        self.renderer.queue.submit([self.encoder.finish()]);
        self.pass.belt.recall();
    }
}

struct GeometryRenderPass<'a> {
    gpu: &'a Renderer,
    rpass: wgpu::RenderPass<'a>,
    current_kind: BatchedGeometryDrawKind,
}

impl GeometryRenderPass<'_> {
    fn update(&mut self, batch: &BatchedGeometryDraw) {
        if batch.kind != self.current_kind {
            self.current_kind = batch.kind;
            match self.current_kind {
                BatchedGeometryDrawKind::Transparent(crate::Transparency::Subtract) => {
                    self.rpass.set_pipeline(&self.gpu.render_pipelines.subtract);
                }
                BatchedGeometryDrawKind::Transparent(
                    crate::Transparency::Add(_) | crate::Transparency::Average,
                )
                | BatchedGeometryDrawKind::OpaqueOnly => {
                    self.rpass.set_pipeline(&self.gpu.render_pipelines.add);
                }
            }
        }
    }

    fn draw_batch(&mut self, batch: &BatchedGeometryDraw) {
        let byte_start = batch.span.start;
        let byte_end = batch.span.end;
        let vertex_start = byte_start / size_of::<Vertex>() as u32;
        let vertex_end = byte_end / size_of::<Vertex>() as u32;

        self.update(batch);

        if let BatchedGeometryDrawKind::Transparent(Transparency::Subtract) = batch.kind {
            self.rpass.set_pipeline(&self.gpu.render_pipelines.subtract);
            self.rpass.draw(vertex_start..vertex_end, 0..1);
            self.rpass
                .set_pipeline(&self.gpu.render_pipelines.subtract_opaque);
            self.rpass.draw(vertex_start..vertex_end, 0..1);
        } else {
            self.rpass.draw(vertex_start..vertex_end, 0..1);
        }
    }
}

struct CpuToVramPass<'a, A: Allocator> {
    batch: &'a CpuToVramBlitBatch<A>,
    cpass: ComputePass<'a>,
}

impl<A: Allocator> CpuToVramPass<'_, A> {
    fn cpu_to_vram_blit(&mut self, blit: &CpuToVramBlit) {
        let uniforms = &self.batch.uniforms[blit.index as usize];
        let dispatch_x = uniforms.size.x.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = uniforms.size.y.div_ceil(WORKGROUP_SIZE);

        if tracing::enabled!(Level::TRACE) {
            tracing::trace!(?uniforms);
        }
        self.cpass.set_immediates(0, &blit.cursor.to_le_bytes());
        self.cpass.set_immediates(4, &blit.index.to_le_bytes());
        self.cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn cpu_to_vram_blit_end(self) {}
}
