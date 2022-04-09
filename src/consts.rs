use glam::Vec3;
use image::{ImageBuffer, Rgba};

pub fn red() -> Vec3 {
    Vec3::new(1.0, 0.0, 0.0)
}

pub fn white() -> Vec3 {
    Vec3::new(1.0, 1.0, 1.0)
}

pub fn green() -> Vec3 {
    Vec3::new(0.0, 1.0, 0.0)
}

pub fn black() -> Vec3 {
    Vec3::new(0.0, 0.0, 0.0)
}

pub const RENDERS: &[for<'r> fn(&'r mut ImageBuffer<Rgba<u8>, Vec<u8>>)] = &[
    super::lesson1::render,
    super::lesson2::render1,
    super::lesson2::render2,
    super::lesson3::triangles,
    super::lesson3_5::triangles,
    super::lesson3_6::triangles,
    super::lesson3_6::triangles2,
];

pub const HEAD_OBJ_BYTES: &[u8] = include_bytes!("head.obj");
