use image::RgbaImage;
use itertools::Itertools;
use std::mem::size_of;
use wgpu::{Buffer, BufferView, Device, SubmissionIndex};

pub fn render(img: &mut RgbaImage) {
    pollster::block_on(run(img));
}

async fn run(img: &mut RgbaImage) {
    let (device, buffer, _, submission_index) =
        create_red_image_with_dimensions(img.width() as usize, img.height() as usize).await;
    create_png(img, device, buffer, submission_index).await;
}

async fn create_red_image_with_dimensions(
    width: usize,
    height: usize,
) -> (Device, Buffer, BufferDimensions, SubmissionIndex) {
    let adapter = wgpu::Instance::new(
        wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all),
    )
    .request_adapter(&wgpu::RequestAdapterOptions::default())
    .await
    .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
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
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (buffer_dimensions.padded_bytes_per_row() * buffer_dimensions.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let texture_extent = wgpu::Extent3d {
        width: buffer_dimensions.width as u32,
        height: buffer_dimensions.height as u32,
        depth_or_array_layers: 1,
    };

    // The render pipeline renders data into this texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        label: None,
    });

    // Set the background to be red
    let command_buffer = {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        // Copy the data from the texture to the buffer
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(buffer_dimensions.padded_bytes_per_row() as u32)
                            .unwrap(),
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

async fn create_png(
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

    fn padded_bytes_per_row(&self) -> usize {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let bytes_per_pixel = size_of::<u32>();
        let unpadded_bytes_per_row = self.width * bytes_per_pixel;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        unpadded_bytes_per_row + padded_bytes_per_row_padding
    }

    fn rows_no_padding<'a>(&self, padded_buffer: &'a BufferView) -> impl Iterator<Item = &'a [u8]> {
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
