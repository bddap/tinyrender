use flate2::read::GzDecoder;
use image::RgbaImage;
use itertools::Itertools;
use std::{borrow::Cow, io::Cursor, mem::size_of, num::NonZeroU32};
use tinyrender::{Atom, Element};
use wgpu::{
    util::DeviceExt, Backends, BlendState, Buffer, BufferDescriptor, BufferUsages, BufferView,
    Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, CompareFunction,
    DepthBiasState, DepthStencilState, Device, DeviceDescriptor, Extent3d, Face, Features,
    FragmentState, FrontFace, ImageCopyBuffer, ImageDataLayout, Instance, Limits, LoadOp,
    MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    StencilState, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, VertexState,
};

pub fn render(img: &mut RgbaImage) {
    pollster::block_on(render_async(img));
}

async fn render_async(img: &mut RgbaImage) {
    let mut pipeline = Pipeline::create(img.width() as usize, img.height() as usize).await;
    pipeline.upload_data(&read_atoms());
    pipeline.render(img).await;
}

fn read_atoms() -> Vec<InstanceInput> {
    let mut reader = GzDecoder::new(Cursor::new(include_bytes!("2btv.bin.gz")));

    let atoms: Vec<Atom> = bincode::deserialize_from(&mut reader).unwrap();

    let coords = atoms.iter().map(|a| a.pos);
    let mut bounds = Bounds::new(coords.clone().next().unwrap_or_default());
    for c in coords {
        bounds.include(c);
    }
    let longest_side = bounds.longest_side();
    let center = bounds.center();

    atoms
        .iter()
        .map(|atom| {
            let x = (atom.pos[0] - center[0]) / longest_side * 2.0;
            let y = (atom.pos[1] - center[1]) / longest_side * 2.0;
            let z = (atom.pos[2] - center[2]) / longest_side * 2.0;

            InstanceInput {
                loc: [x, y, z],
                scale: 0.004,
                color: match atom.element {
                    Element::H => [255, 255, 255],
                    Element::C => [144, 144, 144],
                    Element::O => [255, 13, 13],
                    Element::N => [48, 80, 248],
                    Element::P => [255, 128, 0],
                    Element::S => [255, 255, 48],
                    Element::Na => [171, 92, 242],
                    Element::Mg => [138, 255, 0],
                    Element::Cl => [31, 240, 31],
                    Element::K => [143, 64, 212],
                    Element::Ca => [61, 255, 0],
                    Element::Fe => [224, 102, 51],
                    Element::Mn => [156, 122, 199],
                    Element::Co => [240, 144, 160],
                    Element::Cr => [138, 153, 199],
                    Element::I => [80, 208, 80],
                    Element::Zn => [125, 128, 176],
                    Element::Cu => [200, 128, 51],
                    Element::F => [144, 224, 80],
                    Element::Al => [191, 166, 166],
                    Element::Se => [255, 161, 0],
                    Element::V => [166, 166, 171],
                    Element::Unknown => [255 / 2, 255 / 2, 255 / 2],
                }
                .map(|u: u8| u as f32 / 255.0 / 2.0),
            }
        })
        .collect()
}

struct Bounds {
    max_x: f32,
    max_y: f32,
    max_z: f32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
}

impl Bounds {
    fn new(p: [f32; 3]) -> Self {
        Self {
            max_x: p[0],
            max_y: p[1],
            max_z: p[2],
            min_x: p[0],
            min_y: p[1],
            min_z: p[2],
        }
    }

    fn include(&mut self, p: [f32; 3]) {
        self.max_x = self.max_x.max(p[0]);
        self.min_x = self.min_x.min(p[0]);
        self.max_y = self.max_y.max(p[1]);
        self.min_y = self.min_y.min(p[1]);
        self.max_z = self.max_z.max(p[2]);
        self.min_z = self.min_z.min(p[2]);
    }

    fn center(&self) -> [f32; 3] {
        [
            (self.max_x + self.min_x) / 2.0,
            (self.max_y + self.min_y) / 2.0,
            (self.max_z + self.min_z) / 2.0,
        ]
    }

    fn longest_side(&self) -> f32 {
        (self.max_x - self.min_x)
            .max(self.max_y - self.min_y)
            .max(self.max_z - self.min_z)
    }
}

struct Pipeline {
    device: Device,
    queue: Queue,
    buffer_dimensions: BufferDimensions,
    output_buffer: Buffer,
    texture_extent: Extent3d,
    texture: Texture,
    render_pipeline: RenderPipeline,
    instance_buffer: Buffer,
    instance_count: u32,
    index_buffer: Buffer,
    depth_texture: Texture,
}

impl Pipeline {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    async fn create(width: usize, height: usize) -> Self {
        let adapter =
            Instance::new(wgpu::util::backend_bits_from_env().unwrap_or_else(Backends::all))
                .request_adapter(&RequestAdapterOptions::default())
                .await
                .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: Features::DEPTH_CLIP_CONTROL,
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
                buffers: &[InstanceInput::desc()],
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
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: true,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 3, 4]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_count: u32 = 0;

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
        });

        Self {
            device,
            queue,
            output_buffer,
            buffer_dimensions,
            texture_extent,
            texture,
            render_pipeline,
            instance_buffer,
            instance_count,
            index_buffer,
            depth_texture,
        }
    }

    fn upload_data(&mut self, data: &[InstanceInput]) {
        self.instance_buffer.destroy();
        self.instance_count = data.len().try_into().unwrap();
        self.instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX,
            });
    }

    async fn render(&self, img: &mut RgbaImage) {
        let command_buffer = {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            let texture_view = self.texture.create_view(&TextureViewDescriptor::default());

            let depth_texture_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

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
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_texture_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(f32::NEG_INFINITY),
                            store: false,
                        }),
                        stencil_ops: None,
                    }),
                });

                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

                render_pass.draw_indexed(0..5, 0, 0..self.instance_count);
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceInput {
    loc: [f32; 3],
    scale: f32,
    color: [f32; 3],
}

impl InstanceInput {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: wgpu::VertexFormat::Float32x3.size(),
                shader_location: 1,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: wgpu::VertexFormat::Float32x3.size() + wgpu::VertexFormat::Float32.size(),
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
        ];
        wgpu::VertexBufferLayout {
            array_stride: core::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRIBUTES,
        }
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
