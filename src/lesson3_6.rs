use std::io::Cursor;

use crate::common::to_pixel;

use glam::{UVec2, Vec2, Vec3, Vec3Swizzles};
use image::RgbaImage;
use obj::Obj;

pub fn triangles(img: &mut RgbaImage) {
    let (x, y) = img.dimensions();
    assert_eq!(x, y);
    render(x, |coords, color| {
        img.put_pixel(coords.x, coords.y, to_pixel(color));
    });
}

pub fn render(image_size: u32, mut put_pixel: impl FnMut(UVec2, Vec3)) {
    let obj: Obj = obj::load_obj(Cursor::new(include_bytes!(
        "./obj/african_head/african_head.obj"
    )))
    .unwrap();

    let model_to_screen = |vert: Vec3| -> UVec2 {
        let ret: Vec2 =
            (vert.xy() + Vec2::new(1.0, 1.0)) * Vec2::new(1.0, 1.0) * image_size as f32 / 2.0;
        ret.as_uvec2().min(UVec2::ONE * (image_size - 1))
    };

    let light_dir = Vec3::new(0.0, 1.0, -1.0).normalize();

    for face in faces(&obj) {
        let luma: Vec3 = normal(face) * light_dir;
        let verts = face.map(model_to_screen);
        filled_triangle(&mut put_pixel, verts, random_color().normalize() * luma);
    }
}

pub fn triangles2(img: &mut RgbaImage) {
    let (x, y) = img.dimensions();
    assert_eq!(x, y);
    render2(x, |coords, color| {
        img.put_pixel(coords.x, coords.y, to_pixel(color));
    });
}

pub fn render2(image_size: u32, mut put_pixel: impl FnMut(UVec2, Vec3)) {
    let obj: Obj = obj::load_obj(Cursor::new(include_bytes!(
        "./obj/african_head/african_head.obj"
    )))
    .unwrap();

    let model_to_screen = |vert: Vec3| -> UVec2 {
        let ret: Vec2 =
            (vert.xy() + Vec2::new(1.0, 1.0)) * Vec2::new(1.0, 1.0) * image_size as f32 / 2.0;
        ret.as_uvec2().min(UVec2::ONE * (image_size - 1))
    };

    let light_dir = Vec3::new(0.0, 0.0, -1.0).normalize();

    let color = Vec3::new(0.90478027, 0.31203088, 0.28984386);

    for face in faces(&obj) {
        let luma = normal(face).dot(light_dir);
        if luma > 0.0 {
            let verts = face.map(model_to_screen);
            filled_triangle(&mut put_pixel, verts, color * luma);
        }
    }
}

fn faces<'a>(obj: &'a Obj) -> impl Iterator<Item = [Vec3; 3]> + 'a {
    for index in &obj.indices {
        assert!((*index as usize) < obj.vertices.len());
    }
    obj.indices.chunks(3).map(|c| {
        let a = obj.vertices[c[0] as usize].position;
        let b = obj.vertices[c[1] as usize].position;
        let c = obj.vertices[c[2] as usize].position;
        [a.into(), b.into(), c.into()]
    })
}

/// this function uses magic to to something useful
fn barycentric(pts: [Vec2; 3], p: Vec2) -> Vec3 {
    let u = Vec3::new(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - p[0]).cross(Vec3::new(
        pts[2].y - pts[0].y,
        pts[1].y - pts[0].y,
        pts[0].y - p[1],
    ));
    // `pts` and `P` has integer value as coordinates
    // so `abs(u.z)` < 1 means `u.z` is 0, that means
    // triangle is degenerate, in this case return something with negative coordinates
    if u.z.abs() < 1.0 {
        Vec3::new(-1.0, 1.0, 1.0)
    } else {
        Vec3::new(1.0 - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z)
    }
}

fn filled_triangle(put_pixel: &mut impl FnMut(UVec2, Vec3), verts: [UVec2; 3], color: Vec3) {
    let xmin = verts.iter().map(|vert| vert.x).min().unwrap();
    let xmax = verts.iter().map(|vert| vert.x).max().unwrap();
    let ymin = verts.iter().map(|vert| vert.y).min().unwrap();
    let ymax = verts.iter().map(|vert| vert.y).max().unwrap();
    let verts = verts.map(|v| v.as_vec2());
    for x in xmin..xmax {
        for y in ymin..ymax {
            let ploc = UVec2::new(x, y);
            let bar = barycentric(verts, ploc.as_vec2());
            if bar.x.min(bar.y).min(bar.z) >= 0.0 {
                put_pixel(ploc, color);
            }
        }
    }
}

fn random_color() -> Vec3 {
    Vec3::new(rand::random(), rand::random(), rand::random())
}

fn normal(triangle: [Vec3; 3]) -> Vec3 {
    (triangle[2] - triangle[0])
        .cross(triangle[1] - triangle[0])
        .normalize()
}
