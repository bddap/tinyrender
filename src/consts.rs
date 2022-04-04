use glam::Vec4;
use image::{ImageBuffer, Rgba};

pub fn red() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

pub fn white() -> Vec4 {
    Vec4::new(1.0, 1.0, 1.0, 1.0)
}

pub fn black() -> Vec4 {
    Vec4::new(0.0, 0.0, 0.0, 1.0)
}

pub const RENDERS: &[for<'r> fn(&'r mut ImageBuffer<Rgba<u8>, Vec<u8>>)] = &[
    super::lesson1::render,
    super::lesson2::render1,
    super::lesson2::render2,
];
