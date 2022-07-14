use image::RgbaImage;
use itertools::Itertools;
use std::{borrow::Cow, mem::size_of, num::NonZeroU32};
use wgpu::{Buffer, BufferView, Device, SubmissionIndex};

pub fn render(img: &mut RgbaImage) {
    pollster::block_on(render_async(img));
}

async fn render_async(img: &mut RgbaImage) {
    let (device, buffer, _, submission_index) =
        create_red_image_with_dimensions(img.width() as usize, img.height() as usize).await;
    copy_img(img, device, buffer, submission_index).await;
}

async fn create_red_image_with_dimensions(
    width: usize,
    height: usize,
) -> (Device, Buffer, BufferDimensions, SubmissionIndex) {
    use wgpu::*;

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

    // It is a WebGPU requirement that ImageCopyBuffer.layout.bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
    // So we calculate padded_bytes_per_row by rounding unpadded_bytes_per_row
    // up to the next multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
    // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
    let buffer_dimensions = BufferDimensions::new(width, height);
    // The output buffer lets us retrieve the data as an array
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

    // The render pipeline renders data into this texture
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
            entry_point: "vs_main", // 1.
            buffers: &[],           // 2.
        },
        fragment: Some(FragmentState {
            // 3.
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                // 4.
                format: TextureFormat::Rgba8UnormSrgb,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList, // 1. // can maybe change to PointList
            strip_index_format: None,
            front_face: FrontFace::Ccw, // 2.
            cull_mode: Some(Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None, // 1.
        multisample: MultisampleState {
            count: 1,                         // 2.
            mask: !0,                         // 3.
            alpha_to_coverage_enabled: false, // 4.
        },
        multiview: None, // 5.
    });

    let command_buffer = {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
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

            render_pass.set_pipeline(&render_pipeline); // 2.
            render_pass.draw(0..3, 0..1); // 3.
        }

        // Copy the data from the texture to the buffer
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            ImageCopyBuffer {
                buffer: &output_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        NonZeroU32::new(buffer_dimensions.padded_bytes_per_row() as u32).unwrap(),
                    ),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );

        encoder.finish()
    };

    let index = queue.submit(Some(command_buffer));
    (device, output_buffer, buffer_dimensions, index)
}

/// move image from output_buffer into img
async fn copy_img(
    img: &mut RgbaImage,
    device: Device,
    output_buffer: Buffer,
    submission_index: SubmissionIndex,
) {
    // Note that we're not calling `.await` here.
    let buffer_slice = output_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    //
    // We pass our submission index so we don't need to wait for any other possible submissions.
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

    receiver.receive().await.unwrap().unwrap();

    // copy
    {
        let bytes_per_pixel = 4;
        let unpadded_bytes_per_row = img.width() as usize * bytes_per_pixel;

        let padded_buffer = buffer_slice.get_mapped_range();
        let inrows = BufferDimensions::new(img.width() as usize, img.height() as usize)
            .rows_no_padding(&padded_buffer);

        let samp = img.as_flat_samples_mut().samples;
        let outrows = samp.chunks_mut(unpadded_bytes_per_row);

        for (inrow, outrow) in inrows.zip_eq(outrows) {
            outrow.copy_from_slice(inrow);
        }

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(padded_buffer);
        output_buffer.unmap();
    }
}

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

    fn rows_no_padding<'a>(&self, padded_buffer: &'a BufferView) -> impl Iterator<Item = &'a [u8]> {
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
    use wgpu::BufferView;

    #[test]
    fn ensure_generated_data_matches_expected() {
        pollster::block_on(assert_generated_data_matches_expected());
    }

    async fn assert_generated_data_matches_expected() {
        let (device, output_buffer, dimensions, _) =
            create_red_image_with_dimensions(100usize, 200usize).await;
        let buffer_slice = output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::Wait);
        let padded_buffer = buffer_slice.get_mapped_range();
        let expected_buffer_size = dimensions.padded_bytes_per_row() * dimensions.height;
        assert_eq!(padded_buffer.len(), expected_buffer_size);
        assert_that_content_is_all_red(&dimensions, padded_buffer);
    }

    fn assert_that_content_is_all_red(dimensions: &BufferDimensions, padded_buffer: BufferView) {
        let red = [0xFFu8, 0, 0, 0xFFu8];
        let single_rgba = 4;
        dimensions
            .rows_no_padding(&padded_buffer)
            .flat_map(|unpadded_row| unpadded_row.chunks(single_rgba))
            .for_each(|chunk| assert_eq!(chunk, &red));
    }
}
