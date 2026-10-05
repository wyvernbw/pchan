use alloc::sync::Arc;
use core::cell::Cell;
#[cfg(target_os = "macos")]
use core_foundation::base::TCFType;
#[cfg(target_os = "macos")]
use core_video::pixel_buffer::CVPixelBuffer;
use gpui::prelude::*;
use gpui::*;
use pchan_gpu::wgpu::{self};

#[derive(Debug, Clone)]
pub struct GameSurface {
    id:        ElementId,
    style:     StyleRefinement,
    renderer:  Arc<pchan_gpu::Renderer>,
    pub state: Entity<SurfaceState>,
}

impl GameSurface {
    pub fn new(
        id: impl Into<ElementId>,
        gpu: Arc<pchan_gpu::Renderer>,
        state: Entity<SurfaceState>,
    ) -> Self {
        GameSurface {
            id: id.into(),
            style: StyleRefinement::default(),
            renderer: gpu,
            state,
        }
    }

    pub fn set_vram_view(&self, value: bool) {
        self.renderer.display_uniforms.lock().unwrap().app.dp_debug = value
    }

    pub fn clear<T>(&self, cx: &Context<'_, T>) {
        let state = self.state.read(cx);
        state.clear(&self.renderer);
    }
}

pub struct SurfaceState {
    pub target:         PchanTexture,
    pub target_buf:     wgpu::Buffer,
    rendered_once:      Cell<bool>,
    display_submission: Cell<Option<wgpu::SubmissionIndex>>,

    #[cfg(target_os = "macos")]
    metal_ypcbcr: Option<MetalYpCbCrComputeState>,
}

impl SurfaceState {
    #[must_use]
    pub fn new(target: PchanTexture, target_buf: wgpu::Buffer) -> Self {
        Self {
            target,
            target_buf,
            display_submission: Cell::new(None),
            metal_ypcbcr: None,
            rendered_once: Cell::new(false),
        }
    }

    pub fn clear(&self, gpu: &pchan_gpu::Renderer) {
        use wgpu::*;

        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let range = &ImageSubresourceRange {
            aspect:            TextureAspect::All,
            base_mip_level:    0,
            mip_level_count:   None,
            base_array_layer:  0,
            array_layer_count: None,
        };
        encoder.clear_texture(&self.target.wgpu, range);

        #[cfg(target_os = "macos")]
        {
            let buf = &self.target.metal.pixel_buffer;
            let ret = buf.lock_base_address(0);
            if ret == 0 {
                let y_value = 0;
                unsafe {
                    // Plane 0: luma

                    use core::ptr;

                    let y_base = buf.get_base_address_of_plane(0).cast::<u8>();
                    let y_stride = buf.get_bytes_per_row_of_plane(0);
                    let y_height = buf.get_height_of_plane(0);
                    ptr::write_bytes(y_base, y_value, y_stride * y_height);

                    // Plane 1: interleaved CbCr, half height, 2 bytes per chroma sample.
                    // 128 for both Cb and Cr means neutral chroma, so a single byte value works.
                    let c_base = buf.get_base_address_of_plane(1).cast::<u8>();
                    let c_stride = buf.get_bytes_per_row_of_plane(1);
                    let c_height = buf.get_height_of_plane(1);
                    ptr::write_bytes(c_base, 128, c_stride * c_height);
                }

                buf.unlock_base_address(0);
            }
        }

        let cmd_buf = encoder.finish();
        let sub = gpu.queue.submit([cmd_buf]);
        gpu.device
            .poll(wgt::PollType::Wait {
                submission_index: Some(sub),
                timeout:          None,
            })
            .unwrap();
    }
}

impl IntoElement for GameSurface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for GameSurface {
    type RequestLayoutState = DivFrameState;
    type PrepaintState = ();

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let mut state = self.state.as_mut(cx);
        state.wait_for_display_draw(&self.renderer);
        #[cfg(target_os = "macos")]
        {
            let compute = state
                .metal_ypcbcr
                .take()
                .unwrap_or_else(|| MetalYpCbCrComputeState::new(&self.renderer, &state.target));
            compute.wait_for_conversion(&self.renderer);
            state.metal_ypcbcr = Some(compute);
        }

        if state.rendered_once.get() {
            state.target.draw_into(window, bounds);
        }
    }

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut div = div();
        *div.interactivity().base_style = self.style.clone();

        Element::request_layout(&mut div, id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let width = bounds.size.width;
        let height = bounds.size.height;
        let width = width.as_f32() as u16;
        let height = height.as_f32() as u16;
        let mut dp = self.renderer.display_uniforms.lock().unwrap();

        if width != dp.app.screen_rect.x || height != dp.app.screen_rect.y {
            dp.app.screen_rect.x = width;
            dp.app.screen_rect.y = height;
            let dp = dp.clone();
            let (target, target_buf) = create_target(&self.renderer, &dp);
            self.state.update(cx, |state, _| {
                state.target = target;
                state.target_buf = target_buf;
            });
        };

        // Element::prepaint(
        //     &mut self.div,
        //     id,
        //     inspector_id,
        //     bounds,
        //     request_layout,
        //     window,
        //     cx,
        // )
    }
}

impl Styled for GameSurface {
    #[doc = " Returns a reference to the style memory of this element."]
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

#[derive(Clone)]
pub struct PchanTexture {
    pub wgpu: wgpu::Texture,
    #[cfg(target_os = "macos")]
    metal:    PchanMetalTexture,
}

use core::mem;

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PchanMetalTexture {
    _gpu:              Arc<pchan_gpu::Renderer>,
    pixel_buffer:      CVPixelBuffer,
    dest_yp_texture:   mem::ManuallyDrop<wgpu::Texture>,
    dest_cbcr_texture: mem::ManuallyDrop<wgpu::Texture>,
}

pub fn create_target(
    gpu: &Arc<pchan_gpu::Renderer>,
    dp: &pchan_gpu::DisplayUniforms,
) -> (PchanTexture, wgpu::Buffer) {
    let width = dp.app.screen_rect.x * 4;
    let width = (width + 255) & !255; // align to 256

    #[cfg(not(target_os = "macos"))]
    let target = create_target_texture_linux(gpu, dp);
    #[cfg(target_os = "macos")]
    let target = create_target_texture_metal(gpu, dp);

    let target_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:              None,
        size:               u64::from(width) * u64::from(dp.app.screen_rect.y),
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (target, target_buf)
}

#[cfg(not(target_os = "macos"))]
pub fn create_target_texture_linux(
    gpu: &pchan_gpu::Renderer,
    dp: &pchan_gpu::DisplayUniforms,
) -> PchanTexture {
    let wgpu = create_target_texture_wgpu(gpu, dp);
    PchanTexture { wgpu }
}

pub fn create_target_texture_wgpu(
    gpu: &pchan_gpu::Renderer,
    dp: &pchan_gpu::DisplayUniforms,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label:           None,
        size:            wgpu::Extent3d {
            width:                 u32::from(dp.app.screen_rect.x),
            height:                u32::from(dp.app.screen_rect.y.max(16)),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Bgra8UnormSrgb,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
        ],
    })
}

#[cfg(target_os = "macos")]
pub fn create_target_texture_metal(
    gpu: &Arc<pchan_gpu::Renderer>,
    dp: &pchan_gpu::DisplayUniforms,
) -> PchanTexture {
    use core_foundation::boolean::*;
    use core_foundation::dictionary::*;
    use core_foundation::string::*;

    use core_video::*;
    use objc2_metal::*;

    let width = dp.app.screen_rect.x as usize & !1;
    let height = dp.app.screen_rect.y as usize & !1;

    unsafe {
        use objc2_metal::MTLDevice;

        let attributes = CFDictionary::from_CFType_pairs(&[(
            CFString::wrap_under_get_rule(
                pixel_buffer::kCVPixelBufferMetalCompatibilityKey
                    .as_ref()
                    .unwrap(),
            ),
            CFBoolean::true_value().as_CFType(),
        )]);

        let pixel_buffer = CVPixelBuffer::new(
            pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            width,
            height,
            Some(&attributes),
        )
        .expect("failed to create CVPixelBuffer");

        let metal_device = &*gpu
            .device
            .as_hal::<wgpu::hal::api::Metal>()
            .expect("failed to get metal device");
        let metal_device = metal_device.raw_device();
        let io_surface = pixel_buffer
            .get_io_surface()
            .expect("failed to get IOSurface");
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setWidth(width);
        descriptor.setHeight(height);
        descriptor.setPixelFormat(MTLPixelFormat::R8Unorm);
        let dest_yp_texture = metal_device
            .newTextureWithDescriptor_iosurface_plane(
                &descriptor,
                io_surface
                    .as_concrete_TypeRef()
                    .cast::<objc2_io_surface::IOSurfaceRef>()
                    .as_ref_unchecked(),
                0,
            )
            .expect("failed to create MTLTexture");
        descriptor.setWidth(width / 2);
        descriptor.setHeight(height / 2);
        descriptor.setPixelFormat(MTLPixelFormat::RG8Unorm);
        let dest_cbcr_texture = metal_device
            .newTextureWithDescriptor_iosurface_plane(
                &descriptor,
                io_surface
                    .as_concrete_TypeRef()
                    .cast::<objc2_io_surface::IOSurfaceRef>()
                    .as_ref_unchecked(),
                1,
            )
            .expect("failed to create MTLTexture");

        let wgpu_texture = create_target_texture_wgpu(gpu, dp);
        let wgpu_hal_yp_tex = wgpu::hal::metal::Device::texture_from_raw(
            dest_yp_texture,
            wgpu::TextureFormat::R8Unorm,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width:  width as _,
                height: height as _,
                depth:  1,
            },
            None,
        );
        let wgpu_yp_texture = gpu.device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            wgpu_hal_yp_tex,
            &wgpu::TextureDescriptor {
                label:           None,
                size:            wgpu::Extent3d {
                    width:                 u32::from(dp.app.screen_rect.x),
                    height:                u32::from(dp.app.screen_rect.y.max(16)),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::R8Unorm,
                usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats:    &[wgpu::TextureFormat::R8Unorm],
            },
            wgpu::wgt::TextureUses::STORAGE_WRITE_ONLY,
        );

        let wgpu_hal_cbcr_tex = wgpu::hal::metal::Device::texture_from_raw(
            dest_cbcr_texture,
            wgpu::TextureFormat::Rg8Unorm,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width:  width as u32 / 2,
                height: height as u32 / 2,
                depth:  1,
            },
            None,
        );
        let wgpu_cbcr_texture = gpu.device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            wgpu_hal_cbcr_tex,
            &wgpu::TextureDescriptor {
                label:           None,
                size:            wgpu::Extent3d {
                    width:                 width as u32 / 2,
                    height:                height as u32 / 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rg8Unorm,
                usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats:    &[wgpu::TextureFormat::Rg8Unorm],
            },
            wgpu::wgt::TextureUses::STORAGE_WRITE_ONLY,
        );

        PchanTexture {
            wgpu:  wgpu_texture,
            metal: PchanMetalTexture {
                _gpu: gpu.clone(),
                pixel_buffer,
                dest_yp_texture: mem::ManuallyDrop::new(wgpu_yp_texture),
                dest_cbcr_texture: mem::ManuallyDrop::new(wgpu_cbcr_texture),
            },
        }
    }
}

impl Drop for PchanMetalTexture {
    fn drop(&mut self) {
        // SAFETY: both textures are backed by the same buffer, so dropping
        // one of them should be enough.
        unsafe {
            mem::ManuallyDrop::drop(&mut self.dest_yp_texture);
        };
    }
}

struct MetalYpCbCrComputeState {
    cpipe:             wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    src_sampler:       wgpu::Sampler,
    submission:        Cell<Option<wgpu::SubmissionIndex>>,
}

impl MetalYpCbCrComputeState {
    #[cfg(target_os = "macos")]
    pub fn new(gpu: &pchan_gpu::Renderer, _tex: &PchanTexture) -> Self {
        use wgpu::*;
        let shader = gpu
            .device
            .create_shader_module(include_wgsl!("./convert-ypcbcr.wgsl"));
        let bind_group_layout = gpu
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label:   Some("ypcbcr-convert-bind-group-layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding:    0,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Texture {
                            sample_type:    TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count:      None,
                    },
                    BindGroupLayoutEntry {
                        binding:    1,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count:      None,
                    },
                    BindGroupLayoutEntry {
                        binding:    2,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::StorageTexture {
                            access:         StorageTextureAccess::WriteOnly,
                            format:         TextureFormat::R8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count:      None,
                    },
                    BindGroupLayoutEntry {
                        binding:    3,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::StorageTexture {
                            access:         StorageTextureAccess::WriteOnly,
                            format:         TextureFormat::Rg8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count:      None,
                    },
                ],
            });
        let pipe_layout = gpu
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label:              Some("ypcbcr-convert-pipeline-layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size:     0,
            });
        let cpipe = gpu
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label:               Some("ypcbcr-convert-compute-pipeline"),
                layout:              Some(&pipe_layout),
                module:              &shader,
                entry_point:         None,
                compilation_options: PipelineCompilationOptions {
                    constants:                        &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache:               None,
            });
        let src_texture_sampler = gpu.device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            ..Default::default()
        });
        MetalYpCbCrComputeState {
            cpipe,
            bind_group_layout,
            src_sampler: src_texture_sampler,
            submission: Cell::new(None),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn convert_render(&self, gpu: &pchan_gpu::Renderer, tex: &PchanTexture) {
        use wgpu::*;

        fn view_descriptor<'a>(format: TextureFormat) -> TextureViewDescriptor<'a> {
            TextureViewDescriptor {
                label:             None,
                format:            Some(format),
                dimension:         Some(TextureViewDimension::D2),
                usage:             None,
                aspect:            TextureAspect::All,
                base_mip_level:    0,
                mip_level_count:   None,
                base_array_layer:  0,
                array_layer_count: None,
            }
        }

        let src_texture_view = tex
            .wgpu
            .create_view(&view_descriptor(TextureFormat::Bgra8Unorm));
        let dest_yp_texture_view = tex
            .metal
            .dest_yp_texture
            .create_view(&view_descriptor(TextureFormat::R8Unorm));
        let dest_cbcr_texture_view = tex
            .metal
            .dest_cbcr_texture
            .create_view(&view_descriptor(TextureFormat::Rg8Unorm));

        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label:   Some("ypcbcr-convert-bind-group"),
            layout:  &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding:  0,
                    resource: BindingResource::TextureView(&src_texture_view),
                },
                BindGroupEntry {
                    binding:  1,
                    resource: BindingResource::Sampler(&self.src_sampler),
                },
                BindGroupEntry {
                    binding:  2,
                    resource: BindingResource::TextureView(&dest_yp_texture_view),
                },
                BindGroupEntry {
                    binding:  3,
                    resource: BindingResource::TextureView(&dest_cbcr_texture_view),
                },
            ],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label:            Some("ypcbcr-convert-compute-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.cpipe);
        cpass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 16u32;
        let dispatch_x = tex.wgpu.width().div_ceil(2).div_ceil(workgroup_size);
        let dispatch_y = tex.wgpu.height().div_ceil(2).div_ceil(workgroup_size);

        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(cpass);
        let cmd_buf = encoder.finish();

        let sub = gpu.queue.submit([cmd_buf]);

        self.submission.set(Some(sub));
    }

    pub fn wait_for_conversion(&self, gpu: &pchan_gpu::Renderer) {
        if let Some(sub) = self.submission.take() {
            gpu.device
                .poll(wgpu::wgt::PollType::Wait {
                    submission_index: Some(sub),
                    timeout:          None,
                })
                .unwrap();
        }
    }
}

impl SurfaceState {
    pub fn start_convert_render(&self, gpu: &pchan_gpu::Renderer) {
        #[cfg(target_os = "macos")]
        {
            if let Some(compute) = self.metal_ypcbcr.as_ref() {
                compute.convert_render(gpu, &self.target);
            }
        }
    }
    pub fn wait_for_convert_render(&self, gpu: &pchan_gpu::Renderer) {
        #[cfg(target_os = "macos")]
        {
            if let Some(compute) = self.metal_ypcbcr.as_ref() {
                compute.wait_for_conversion(gpu);
            }
        }
        self.rendered_once.set(true);
    }
    pub fn start_display_draw(&self, gpu: &pchan_gpu::Renderer) {
        let sub = draw_display(gpu, &self.target.wgpu, &self.target_buf);
        self.display_submission.set(Some(sub));
    }
    pub fn wait_for_display_draw(&self, gpu: &pchan_gpu::Renderer) {
        if let Some(sub) = self.display_submission.take() {
            _ = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: Some(sub),
                timeout:          None,
            });
            self.rendered_once.set(true);
        }
    }
}

pub(crate) fn draw_display(
    gpu: &pchan_gpu::Renderer,
    target: &wgpu::Texture,
    target_buf: &wgpu::Buffer,
) -> wgpu::SubmissionIndex {
    use pchan_gpu::wgpu;
    use wgpu::*;

    let target_view = target.create_view(&TextureViewDescriptor {
        label:             None,
        format:            Some(TextureFormat::Bgra8UnormSrgb),
        dimension:         None,
        aspect:            wgpu::TextureAspect::All,
        base_mip_level:    0,
        mip_level_count:   Some(target.mip_level_count()),
        base_array_layer:  0,
        array_layer_count: None,
        usage:             Some(TextureUsages::RENDER_ATTACHMENT),
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
            view:           &target_view,
            resolve_target: None,
            ops:            wgpu::Operations {
                load:  wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice:    None,
        })],
        depth_stencil_attachment: None,
        label:                    Some("display render pass"),
        timestamp_writes:         None,
        occlusion_query_set:      None,
        multiview_mask:           None,
    });
    gpu.draw_display(&mut rpass);
    drop(rpass);
    let unpadded = target.width() * 4;
    let padded = (unpadded + 255) & !255; // align to 256
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: target_buf,
            layout: TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(padded),
                rows_per_image: Some(target.height()),
            },
        },
        target.size(),
    );

    let commands = encoder.finish();

    gpu.queue.submit([commands])
}

impl PchanTexture {
    pub fn draw_into(&self, window: &mut Window, bounds: Bounds<Pixels>) {
        #[cfg(target_os = "macos")]
        {
            window.paint_surface(bounds, self.metal.pixel_buffer.clone());
        }
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        {
            window.paint_surface(bounds, self.wgpu);
        }
    }
}
