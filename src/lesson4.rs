use std::{
    io::Cursor,
    ops::{Add, Mul},
};

use crate::common::to_pixel;

use glam::{Mat4, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use image::{ImageFormat, RgbaImage};
use obj::raw::{object::Polygon, parse_obj};

struct Frame {
    width: usize,
    /// colors + zbuffler
    pix: Vec<Vec4>,
}

impl Frame {
    fn new(width: usize, height: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        let len = width.checked_mul(height).unwrap();
        let fill = [0.0, 0.0, 0.0, f32::NEG_INFINITY].into();
        Self {
            width,
            pix: vec![fill; len],
        }
    }

    fn get_mut(&mut self, x: u32, y: u32) -> &mut Vec4 {
        let (x, y) = (x as usize, y as usize);
        debug_assert!(x < self.width);
        debug_assert!(y < self.pix.len() / self.width);
        self.pix.get_mut(x + y * self.width).unwrap()
    }

    fn write(&self, img: &mut RgbaImage) {
        let (w, h) = img.dimensions();
        assert_eq!(w as usize, self.width);
        assert_eq!(h as usize, self.pix.len() / self.width);
        for (i, c) in self.pix.iter().enumerate() {
            let x = i % self.width;
            let y = i / self.width;
            img.put_pixel(x as u32, y as u32, to_pixel(c.xyz()));
        }
    }
}

pub fn render(img: &mut RgbaImage) {
    let (w, h) = img.dimensions();
    assert_eq!(w, h);
    let mut frame = Frame::new(w as usize, h as usize);
    render_(w, &mut frame);
    frame.write(img);
}

fn render_(image_size: u32, frame: &mut Frame) {
    let tex = texture();
    let texsize = Vec2::new(tex.width() as f32, tex.height() as f32);

    let screen_size = UVec2::splat(image_size);

    let model_to_screen = Mat4::from_scale(Vec3::splat(image_size as f32) / 2.0)
        * Mat4::from_translation(Vec3::splat(1.0));

    // let model_to_screen = perspective * a;

    let to_screen = |vert: Vec3| {
        // I still don't quite understand where perspective usually

        // perspective
        let camera_distance_from_origin = 6.0; // camera is at [0.0, 0.0, c] facing origin
        let div = 1.0 - vert.z / camera_distance_from_origin;
        let vert = vert / div;

        // the rest
        let vert = model_to_screen.transform_point3(vert);
        vert
    };

    let light_dir = Vec3::new(0.0, 0.0, -1.0).normalize();

    for tri in model() {
        let face = tri.map(|(f, _uv)| f);
        let uvs = tri.map(|(_f, uv)| uv);
        let luma: f32 = normal(face).dot(light_dir);
        if luma > 0.0 {
            let verts = face.map(|v| to_screen(v).xy().as_uvec2());
            let zs: Vec3 = face.map(|face| face.z).into();
            for_coord_in_triangle(screen_size, verts, |pos, bar| {
                let pix = frame.get_mut(pos.x, pos.y);
                let z: f32 = bary_interp(bar, zs.to_array());
                let uv: Vec2 = texsize - bary_interp(bar, uvs) * texsize;
                let color: Vec3 = tex.get_pixel(uv.x as u32, uv.y as u32).0.into();
                if pix.w < z {
                    *pix = (color * luma).extend(z);
                }
            });
        }
    }
}

// the way it should be:
// fn bary_interp<T, O>(bar: Vec3, pts: [T; 3]) -> O
// where
//     T: Mul<f32>,
//     O: Sum<T::Output>,
// {
//     bar.to_array()
//         .iter()
//         .zip(pts)
//         .map(|(b, p)| p * *b)
//         .sum::<O>()
// }
// but Vec2 does not implement Sum<Vec2>.
// it implements Sum<&'_ Vec2>

fn bary_interp<T>(bar: Vec3, pts: [T; 3]) -> T
where
    T: Mul<f32, Output = T> + Add<T, Output = T>,
{
    bar.to_array()
        .iter()
        .zip(pts)
        .map(|(b, p)| p * *b)
        .reduce(|a, b| a + b)
        .unwrap()
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

// generic triangle filling algo, provides position and barycentric coords to the callback
fn for_coord_in_triangle(screen: UVec2, verts: [UVec2; 3], mut cb: impl FnMut(UVec2, Vec3)) {
    let xmin = verts.iter().map(|vert| vert.x).min().unwrap();
    let xmax = verts.iter().map(|vert| vert.x).max().unwrap().min(screen.x);
    let ymin = verts.iter().map(|vert| vert.y).min().unwrap();
    let ymax = verts.iter().map(|vert| vert.y).max().unwrap().min(screen.y);
    let verts = verts.map(|v| v.as_vec2());
    for x in xmin..xmax {
        for y in ymin..ymax {
            let ploc = UVec2::new(x, y);
            let bar = barycentric(verts, ploc.as_vec2());
            if bar.x.min(bar.y).min(bar.z) >= 0.0 {
                cb(ploc, bar);
            }
        }
    }
}

fn normal(triangle: [Vec3; 3]) -> Vec3 {
    (triangle[2] - triangle[0])
        .cross(triangle[1] - triangle[0])
        .normalize()
}

// positions and texure coords
fn model() -> Vec<([(Vec3, Vec2); 3])> {
    let obj = parse_obj(Cursor::new(include_bytes!("head.obj"))).unwrap();
    let mut ret = Vec::<[(Vec3, Vec2); 3]>::new();
    for poly in obj.polygons {
        match poly {
            Polygon::P(_) => panic!("Model does not provide texture coords!"),
            Polygon::PT(ref pts) => {
                assert_eq!(pts.len(), 3);
                let face = [pts[0], pts[1], pts[2]];
                let face = face.map(|(p, t)| {
                    (
                        Vec4::from(obj.positions[p]).xyz(),
                        Vec3::from(obj.tex_coords[t]).xy(),
                    )
                });
                ret.push(face);
            }
            Polygon::PN(_) => panic!("Model does not provide texture coords!"),
            Polygon::PTN(ref ptns) => {
                assert_eq!(ptns.len(), 3);
                let face = [ptns[0], ptns[1], ptns[2]];
                let face = face.map(|(p, t, _n)| {
                    (
                        Vec4::from(obj.positions[p]).xyz(),
                        Vec3::from(obj.tex_coords[t]).xy(),
                    )
                });
                ret.push(face);
            }
        };
    }

    if cfg!(debug_assertions) {
        for face in &ret {
            for (_, uv) in face {
                assert!(uv.x <= 1.0);
                assert!(uv.y <= 1.0);
                assert!(uv.x >= 0.0);
                assert!(uv.y >= 0.0);
            }
        }
    }

    ret
}

fn texture() -> image::Rgb32FImage {
    image::load_from_memory_with_format(include_bytes!("head_diffuse.png"), ImageFormat::Png)
        .unwrap()
        .to_rgb32f()
}
