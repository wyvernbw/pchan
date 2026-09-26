use gpui::prelude::*;
use gpui::*;
use kanal::{Receiver, Sender};
use pchan_emu::gpu::Rgb5;
use pchan_gpu::wgpu;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, MutexGuard};

use crate::Debugger;

pub struct GameSurface {
    div:       Div,
    renderer:  Arc<pchan_gpu::Renderer>,
    pub state: Entity<SurfaceState>,
    vram:      Option<Box<[u16]>>,
}

#[derive(Clone)]
pub struct SurfaceState {
    pub target:        wgpu::Texture,
    pub target_buf:    wgpu::Buffer,
    pub fifo_tx:       Sender<Arc<RenderImage>>,
    pub fifo_rx:       Receiver<Arc<RenderImage>>,
    pub buffer_mapped: Arc<AtomicBool>,
}

impl IntoElement for GameSurface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for GameSurface {
    type RequestLayoutState = DivFrameState;
    type PrepaintState = Option<Hitbox>;

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
        let state = self.state.read(cx);
        let img = match self.vram.as_ref() {
            None => get_display_result(&self.renderer, &state.fifo_rx),
            Some(vram) => {
                // TODO: use preallocated buffer
                let mut buf = vec![0u8; 1024 * 512 * 4];
                for (i, pixel) in vram.iter().enumerate() {
                    let pixel = Rgb5::new_with_raw_value(*pixel);
                    buf[i * 4] = ((pixel.b().value() as u16) * 256 / 32) as u8;
                    buf[i * 4 + 1] = ((pixel.g().value() as u16) * 256 / 32) as u8;
                    buf[i * 4 + 2] = ((pixel.r().value() as u16) * 256 / 32) as u8;
                    buf[i * 4 + 3] = 255;
                }
                let frame =
                    image::Frame::new(image::ImageBuffer::from_raw(1024, 512, buf).unwrap());
                Arc::new(RenderImage::new([frame]))
            }
        };
        window
            .paint_image(
                bounds,
                Bounds::new(
                    Point {
                        x: 0.0.into(),
                        y: 0.0.into(),
                    },
                    Size {
                        width:  state.target.width().into(),
                        height: state.target.height().into(),
                    },
                ),
                Corners::default(),
                img,
                0,
                false,
            )
            .unwrap();
    }

    fn id(&self) -> Option<ElementId> {
        Element::id(&self.div)
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        Element::source_location(&self.div)
    }

    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        Element::request_layout(&mut self.div, id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let width = bounds.size.width;
        let height = bounds.size.height;
        let width = width.as_f32() as u16;
        let height = height.as_f32() as u16;
        let mut dp = self.renderer.display_uniforms.lock().unwrap();

        if width != dp.screen_rect.x || height != dp.screen_rect.y {
            dp.screen_rect.x = width;
            dp.screen_rect.y = height;
            let (target, target_buf) = create_target(&self.renderer, &mut dp);
            self.state.update(cx, |state, _| {
                state.target = target;
                state.target_buf = target_buf;
            })
        }

        Element::prepaint(
            &mut self.div,
            id,
            inspector_id,
            bounds,
            request_layout,
            window,
            cx,
        )
    }
}

impl Styled for GameSurface {
    #[doc = " Returns a reference to the style memory of this element."]
    fn style(&mut self) -> &mut StyleRefinement {
        self.div.style()
    }
}

impl Debugger {
    pub fn pchan_game_surface(&mut self, state: Entity<SurfaceState>) -> GameSurface {
        GameSurface {
            div: div(),
            renderer: self.renderer.clone(),
            state,
            vram: None,
        }
    }
}

pub fn create_target(
    gpu: &pchan_gpu::Renderer,
    dp: &mut MutexGuard<'_, pchan_gpu::DisplayUniforms>,
) -> (wgpu::Texture, wgpu::Buffer) {
    let width = dp.screen_rect.x * 4;
    let width = (width + 255) & !255; // align to 256
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label:           None,
        size:            wgpu::Extent3d {
            width:                 dp.screen_rect.x as u32,
            height:                dp.screen_rect.y.max(16) as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Bgra8UnormSrgb,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats:    &[wgpu::TextureFormat::Bgra8UnormSrgb],
    });
    let target_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:              None,
        size:               width as u64 * dp.screen_rect.y as u64,
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (target, target_buf)
}

pub(crate) fn draw_display(
    gpu: &pchan_gpu::Renderer,
    target: &wgpu::Texture,
    target_buf: &wgpu::Buffer,
    fifo_tx: &Sender<Arc<RenderImage>>,
    buffer_mapped: Arc<AtomicBool>,
) {
    use pchan_gpu::wgpu;
    use wgpu::*;

    let target_view = target.create_view(&TextureViewDescriptor {
        label:             None,
        format:            Some(TextureFormat::Bgra8UnormSrgb),
        dimension:         None,
        aspect:            wgpu::TextureAspect::All,
        base_mip_level:    0,
        mip_level_count:   None,
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

    // spin if already mapped
    while buffer_mapped.load(std::sync::atomic::Ordering::Acquire) {}

    gpu.queue.submit([commands]);

    target_buf.map_async(MapMode::Read, .., {
        let fifo_tx = fifo_tx.clone();
        let target_shadow = target_buf.clone();
        let width = target.width();
        let height = target.height();
        let buffer_mapped = buffer_mapped.clone();

        move |_| {
            buffer_mapped.store(true, std::sync::atomic::Ordering::Release);
            let data = target_shadow.get_mapped_range(..).unwrap();
            let mut buf = Vec::with_capacity(data.len());
            for row in data.chunks(padded as usize) {
                buf.extend_from_slice(&row[..unpadded as usize]);
            }

            let frame =
                image::Frame::new(image::ImageBuffer::from_raw(width, height, buf).unwrap());
            let img = RenderImage::new([frame]);
            drop(data);
            target_shadow.unmap();

            buffer_mapped.store(false, std::sync::atomic::Ordering::Release);
            let img = Arc::new(img);
            fifo_tx.send(img).unwrap();
        }
    });
}

fn get_display_result(
    gpu: &pchan_gpu::Renderer,
    fifo_rx: &Receiver<Arc<RenderImage>>,
) -> Arc<RenderImage> {
    use wgpu::*;

    loop {
        match fifo_rx.try_recv_realtime() {
            Ok(Some(frame)) => {
                return frame;
            }
            _ => {
                _ = gpu.device.poll(PollType::Poll);
                std::thread::yield_now();
            }
        }
    }
}
