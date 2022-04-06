use glam::Vec4;
use image::Rgba;

pub fn to_pixel(color: Vec4) -> Rgba<u8> {
    let r = (color.x * (u8::max_value() as f32)) as u8;
    let g = (color.y * (u8::max_value() as f32)) as u8;
    let b = (color.z * (u8::max_value() as f32)) as u8;
    let a = (color.w * (u8::max_value() as f32)) as u8;
    Rgba([r, g, b, a])
}

pub fn invert_color(color: Vec4) -> Vec4 {
    let a = color.w;
    let mut ret = color - color * 2.0 + 1.0;
    ret.w = a;
    ret
}

pub fn darken_color(color: Vec4) -> Vec4 {
    let a = color.w;
    let mut ret = color * 0.5;
    ret.w = a;
    ret
}
