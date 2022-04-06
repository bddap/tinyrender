use std::io::Cursor;

use crate::common::to_pixel;

use super::consts::*;
use glam::{IVec2, UVec2, Vec2, Vec2Swizzles, Vec4};
use image::RgbaImage;
use obj::Obj;

pub fn render1(img: &mut RgbaImage) {
    let (x, y) = img.dimensions();
    assert_eq!(x, y);
    render1_(x, |coords, color| {
        img.put_pixel(coords.x, coords.y, to_pixel(color));
    });
}

pub fn render1_(image_size: u32, mut put_pixel: impl FnMut(UVec2, Vec4)) {
    let lines = &[
        (UVec2::new(13, 20), UVec2::new(80, 40), white()),
        (UVec2::new(20, 13), UVec2::new(40, 80), red()),
        (UVec2::new(80, 40), UVec2::new(13, 20), red()),
    ];

    let scale = UVec2::ONE * image_size / 100;

    for (from, to, color) in lines {
        let from = *from * scale;
        let to = *to * scale;
        debug_assert!(from.x < image_size);
        debug_assert!(to.x < image_size);
        debug_assert!(from.y < image_size);
        debug_assert!(to.y < image_size);
        line(&mut put_pixel, from, to, *color);
    }
}

fn line(mut put_pixel: impl FnMut(UVec2, Vec4), from: UVec2, to: UVec2, color: Vec4) {
    let mut from: IVec2 = from.as_ivec2();
    let mut to: IVec2 = to.as_ivec2();

    let delta = to - from;
    let steep = delta.x.abs() < delta.y.abs();
    if steep {
        from = from.yx();
        to = to.yx();
    }

    if from.x > to.x {
        std::mem::swap(&mut to, &mut from);
    }

    let delta = to - from;

    let derror2 = delta.y.abs() * 2;
    let mut error2 = 0;
    let mut y = from.y;
    for x in (from.x)..to.x {
        if steep {
            put_pixel(UVec2::new(x as u32, y as u32), color);
        } else {
            put_pixel(UVec2::new(y as u32, x as u32), color);
        }
        error2 += derror2;
        if error2 > delta.x {
            if delta.y > 0 {
                y += 1;
            } else {
                y -= 1;
            }
            error2 -= delta.x * 2;
        }
    }
}

pub fn render2(img: &mut RgbaImage) {
    let (x, y) = img.dimensions();
    assert_eq!(x, y);
    render2_(x, |coords, color| {
        img.put_pixel(coords.x, coords.y, to_pixel(color));
    });
}

pub fn render2_(image_size: u32, mut put_pixel: impl FnMut(UVec2, Vec4)) {
    let obj: Obj = obj::load_obj(Cursor::new(include_bytes!("head.obj"))).unwrap();

    let model_to_screen = |mut vert: Vec2| {
        // didn't expect this swizzle to be needed, adding it as a hack
        vert = vert.yx();

        let ret: Vec2 =
            (vert + Vec2::new(1.0, 1.0)) * Vec2::new(1.0, 1.0) * image_size as f32 / 2.0;
        ret.as_uvec2().min(UVec2::ONE * (image_size - 1))
    };

    for face in faces(&obj) {
        let [a, b, c] = face;
        let (a, b, c) = (model_to_screen(a), model_to_screen(b), model_to_screen(c));
        line(&mut put_pixel, a, b, white());
        line(&mut put_pixel, b, c, white());
        line(&mut put_pixel, c, a, white());
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
