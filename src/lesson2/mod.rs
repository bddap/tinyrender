use std::io::Cursor;

use crate::common::to_pixel;

use super::consts::*;
use glam::{UVec2, Vec2, Vec2Swizzles, Vec4};
use image::RgbaImage;
use obj::Obj;

pub fn render1(image: &mut RgbaImage) {
    let (w, h) = image.dimensions();
    assert!(w > 0);
    assert!(h > 0);
    assert!(h == w);

    let lines = &[
        (UVec2::new(13, 20), UVec2::new(80, 40), white()),
        (UVec2::new(20, 13), UVec2::new(40, 80), red()),
        (UVec2::new(80, 40), UVec2::new(13, 20), red()),
    ];

    let scale = UVec2::new(w, w) / 100;

    for (from, to, color) in lines {
        let from = *from * scale;
        let to = *to * scale;
        debug_assert!(from.x < w);
        debug_assert!(to.x < w);
        debug_assert!(from.y < h);
        debug_assert!(to.y < h);
        line(image, from, to, *color);
    }
}

fn line(image: &mut RgbaImage, from: UVec2, to: UVec2, color: Vec4) {
    let (w, h) = image.dimensions();
    debug_assert!(from.x < w);
    debug_assert!(to.x < w);
    debug_assert!(from.y < h);
    debug_assert!(to.y < h);

    let mut from = from;
    let mut to = to;
    let mut swizzle = [1, 0];

    if abs_diff(from.x, to.x) < abs_diff(from.y, to.y) {
        from = from.yx();
        to = to.yx();
        swizzle = [0, 1];
    }

    if from.x > to.x {
        std::mem::swap(&mut from, &mut to);
    }

    debug_assert!(from.x <= to.x);
    debug_assert!(from.y <= to.y);

    let diff = (to - from) * 1000;

    for t in 0..(to.x - from.x) {
        let ydiff = t * diff.y / diff.x;
        let x = from.x + t;
        let y = to.y + ydiff;

        image.put_pixel(
            swizzle[0] * x + swizzle[1] * y,
            swizzle[0] * y + swizzle[1] * x,
            to_pixel(color),
        );
    }
}

fn abs_diff(a: u32, b: u32) -> u32 {
    if a < b {
        b - a
    } else {
        a - b
    }
}

pub fn render2(image: &mut RgbaImage) {
    let (w, h) = image.dimensions();
    assert!(w > 0);
    assert!(h > 0);
    assert!(h == w);

    let obj: Obj = obj::load_obj(Cursor::new(include_bytes!("head.obj"))).unwrap();

    let scale = (Vec2::new(w as f32, w as f32) + 1.0) / 2.0;

    for face in faces(&obj) {
        let [a, b, c] = face;
        let (a, b, c) = (a * scale, b * scale, c * scale);
        let (a, b, c) = (a.as_uvec2(), b.as_uvec2(), c.as_uvec2());
        line(image, a, b, white());
        line(image, b, c, white());
        line(image, c, a, white());
    }
}

fn faces<'a>(obj: &'a Obj) -> impl Iterator<Item = [Vec2; 3]> + 'a {
    for index in &obj.indices {
        assert!((*index as usize) < obj.vertices.len());
    }
    obj.indices.chunks(3).map(|c| {
        let a = obj.vertices[c[0] as usize].position;
        let b = obj.vertices[c[1] as usize].position;
        let c = obj.vertices[c[2] as usize].position;
        let a = Vec2::new(a[0], a[1]);
        let b = Vec2::new(b[0], b[1]);
        let c = Vec2::new(c[0], c[1]);
        [a, b, c]
    })
}
