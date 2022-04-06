use crate::common::{darken_color, to_pixel};

use super::consts::*;
use glam::{IVec2, UVec2, Vec2Swizzles, Vec4};
use image::RgbaImage;

pub fn triangles(img: &mut RgbaImage) {
    let (x, y) = img.dimensions();
    assert_eq!(x, y);
    render(x, |coords, color| {
        img.put_pixel(coords.x, coords.y, to_pixel(color));
    });
}

pub fn render(image_size: u32, mut put_pixel: impl FnMut(UVec2, Vec4)) {
    let p = UVec2::new;
    let test_triangles = [
        ([p(10, 70), p(50, 160), p(70, 80)], red()),
        ([p(180, 50), p(150, 1), p(70, 180)], white()),
        ([p(180, 150), p(120, 160), p(130, 180)], green()),
    ];

    for (verts, color) in test_triangles {
        let verts = verts.map(|v| (v * (image_size - 1) / 200));
        triangle_wireframe(&mut put_pixel, verts, color);
        filled_triangle(&mut put_pixel, verts, darken_color(color));
    }
}

fn triangle_wireframe(put_pixel: &mut impl FnMut(UVec2, Vec4), verts: [UVec2; 3], color: Vec4) {
    line(put_pixel, verts[0], verts[1], color);
    line(put_pixel, verts[1], verts[2], color);
    line(put_pixel, verts[2], verts[0], color);
}

fn line(put_pixel: &mut impl FnMut(UVec2, Vec4), from: UVec2, to: UVec2, color: Vec4) {
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
            put_pixel(UVec2::new(y as u32, x as u32), color);
        } else {
            put_pixel(UVec2::new(x as u32, y as u32), color);
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

fn filled_triangle(put_pixel: &mut impl FnMut(UVec2, Vec4), verts: [UVec2; 3], color: Vec4) {
    let [mut va, mut vb, mut vc] = verts;
    if va.y > vb.y {
        std::mem::swap(&mut va, &mut vb);
    }
    if va.y > vc.y {
        std::mem::swap(&mut va, &mut vc);
    }
    if vb.y > vc.y {
        std::mem::swap(&mut vb, &mut vc);
    }
    debug_assert!(va.y <= vb.y);
    debug_assert!(vb.y <= vc.y);
    debug_check(put_pixel, va, color * 1.5);
    debug_check(put_pixel, vb, color * 1.5);
    debug_check(put_pixel, vc, color * 1.5);
    for yadd in 0..(vc.y - va.y) {
        put_pixel(UVec2::new(va.x, va.y + yadd), color);
    }
}

fn debug_check(put_pixel: &mut impl FnMut(UVec2, Vec4), pos: UVec2, color: Vec4) {
    for i in 0..10 {
        put_pixel(UVec2::new(pos.x, pos.y.saturating_sub(i)), color);
        put_pixel(UVec2::new(pos.x.saturating_sub(i), pos.y), color);
        put_pixel(
            UVec2::new(pos.x.saturating_sub(i), pos.y.saturating_sub(i)),
            color,
        );
    }
}
