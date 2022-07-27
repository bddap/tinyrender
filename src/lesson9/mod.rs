use image::RgbaImage;
use itertools::Itertools;
use std::{borrow::Cow, mem::size_of, num::NonZeroU32};
use wgpu::*;
use wgpu::{Buffer, BufferView, Device};

pub fn render(img: &mut RgbaImage) {
    pollster::block_on(render_async(img));
}

async fn render_async(img: &mut RgbaImage) {
    let pipeline = Pipeline::create(img.width() as usize, img.height() as usize).await;
    pipeline.render(img).await;
}

struct Pipeline {
    device: Device,
    queue: Queue,
    buffer_dimensions: BufferDimensions,
    output_buffer: Buffer,
    texture_extent: Extent3d,
    texture: Texture,
    render_pipeline: RenderPipeline,
}

impl Pipeline {
    async fn create(width: usize, height: usize) -> Self {
        let adapter = Instance::new(util::backend_bits_from_env().unwrap_or_else(Backends::all))
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: Features::empty(),
                    limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let buffer_dimensions = BufferDimensions::new(width, height);
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (buffer_dimensions.padded_bytes_per_row() * buffer_dimensions.height) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_extent = Extent3d {
            width: buffer_dimensions.width as u32,
            height: buffer_dimensions.height as u32,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            label: None,
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            device,
            queue,
            output_buffer,
            buffer_dimensions,
            texture_extent,
            texture,
            render_pipeline,
        }
    }

    async fn render(&self, img: &mut RgbaImage) {
        let command_buffer = {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            let texture_view = self.texture.create_view(&TextureViewDescriptor::default());
            {
                let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color {
                                r: 0.8,
                                g: 0.2,
                                b: 0.2,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.draw(0..3, 0..1);
            }

            encoder.copy_texture_to_buffer(
                self.texture.as_image_copy(),
                ImageCopyBuffer {
                    buffer: &self.output_buffer,
                    layout: ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(
                            NonZeroU32::new(self.buffer_dimensions.padded_bytes_per_row() as u32)
                                .unwrap(),
                        ),
                        rows_per_image: None,
                    },
                },
                self.texture_extent,
            );

            encoder.finish()
        };

        let index = self.queue.submit(core::iter::once(command_buffer));

        let buffer_slice = self.output_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(index));

        receiver.receive().await.unwrap().unwrap();

        let padded_buffer: BufferView = buffer_slice.get_mapped_range();
        copy_img(img, &padded_buffer);

        drop(padded_buffer);
        self.output_buffer.unmap();
    }
}

/// move image from padded_buffer into img
fn copy_img(img: &mut RgbaImage, padded_buffer: &[u8]) {
    let bytes_per_pixel = 4;
    let unpadded_bytes_per_row = img.width() as usize * bytes_per_pixel;

    let inrows = BufferDimensions::new(img.width() as usize, img.height() as usize)
        .rows_no_padding(&padded_buffer);

    let samp = img.as_flat_samples_mut().samples;
    let outrows = samp.chunks_mut(unpadded_bytes_per_row);

    for (inrow, outrow) in inrows.zip_eq(outrows) {
        outrow.copy_from_slice(inrow);
    }
}

/// It is a WebGPU requirement that ImageCopyBuffer.layout.bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
/// So we calculate padded_bytes_per_row by rounding unpadded_bytes_per_row
/// up to the next multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
/// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
struct BufferDimensions {
    width: usize,
    height: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    fn padding(&self) -> usize {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let bytes_per_pixel = size_of::<u32>();
        let unpadded_bytes_per_row = self.width * bytes_per_pixel;
        (align - unpadded_bytes_per_row % align) % align
    }

    fn padded_bytes_per_row(&self) -> usize {
        let bytes_per_pixel = size_of::<u32>();
        self.width * bytes_per_pixel + self.padding()
    }

    fn rows_no_padding<'a>(&self, padded_buffer: &'a [u8]) -> impl Iterator<Item = &'a [u8]> {
        assert_eq!(
            padded_buffer.len(),
            self.padded_bytes_per_row() * self.height
        );

        let bytes_per_pixel = size_of::<u32>();
        let width = self.width;

        padded_buffer
            .chunks(self.padded_bytes_per_row())
            .map(move |padded_row| &padded_row[..width * bytes_per_pixel])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn generated() {
        let mut img = RgbaImage::new(100, 200);
        render(&mut img);
        for pixel in img.pixels() {
            assert!(
                pixel == &Rgba::from([231, 124, 124, 255])
                    || pixel == &Rgba::from([108, 89, 63, 255])
            );
        }
    }
}
