use std::{
    f32::consts::PI,
    io::Cursor,
    ops::{Add, Mul, Neg},
};

use crate::common::to_pixel;

use glam::{Mat4, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use image::{imageops::flip_vertical_in_place, ImageFormat, RgbaImage};
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

    fn height(&self) -> usize {
        self.pix.len() / self.width
    }
}

trait Shader {
    type Intermediate: Mul<f32, Output = Self::Intermediate>
        + Add<Self::Intermediate, Output = Self::Intermediate>
        + Copy;

    fn vertex(&self, vert: Vec3, normal: Vec3, uv: Vec2) -> (Vec3, Self::Intermediate);

    fn fragment(&self, interp: Self::Intermediate) -> Option<Vec3>;

    fn run(&self, target: &mut Frame, model: &[[(Vec3, Vec2, Vec3); 3]]) {
        let screen_size = UVec2::new(target.width as u32, target.height() as u32);

        for tri in model {
            let verts = tri.map(|(v, _, _)| v);
            if normal(verts).z <= 0.0 {
                let vertout = tri.map(|(pos, uv, normal)| self.vertex(pos, normal, uv));
                let zs = vertout.map(|(pos, _)| pos.z);
                let verts = vertout.map(|(pos, _)| pos.xy().as_uvec2());
                let intermediates = vertout.map(|(_, int)| int);

                for_coord_in_triangle(screen_size, verts, |pos, bar| {
                    let z = bary_interp(bar, zs);
                    let pix = target.get_mut(pos.x, pos.y);
                    if pix.w >= z {
                        return;
                    }
                    let intermediate = bary_interp(bar, intermediates.clone());
                    if let Some(color) = self.fragment(intermediate) {
                        *pix = color.extend(z);
                    }
                });
            }
        }
    }
}

struct Gourad {
    model_to_screen: Mat4,
    model_normal_transform: Mat4,
    texture: image::Rgb32FImage,
    normal_map: image::Rgb32FImage,
    texsize: Vec2,
    light_direction: Vec3,
}

impl Gourad {
    fn new(
        model_to_screen: Mat4,
        model_normal_transform: Mat4,
        texture: image::Rgb32FImage,
        normal_map: image::Rgb32FImage,
        light_direction: Vec3,
    ) -> Self {
        let texsize = Vec2::new(texture.width() as f32, texture.height() as f32);
        assert_eq!(
            (texture.width(), texture.height()),
            (normal_map.width(), normal_map.height())
        );
        debug_assert!((light_direction - light_direction.normalize()).length() < 0.0001);
        Self {
            model_to_screen,
            model_normal_transform,
            texture,
            normal_map,
            texsize,
            light_direction,
        }
    }
}

impl Shader for Gourad {
    type Intermediate = GouradIntermediate;

    fn vertex(&self, vert: Vec3, normal: Vec3, uv: Vec2) -> (Vec3, Self::Intermediate) {
        let pos = self.model_to_screen.project_point3(vert);
        let normal = self.model_normal_transform.transform_vector3(normal);
        let uv = uv * self.texsize;
        (pos, GouradIntermediate { normal, uv, pos })
    }

    fn fragment(&self, interp: Self::Intermediate) -> Option<Vec3> {
        let (px, py) = (interp.uv.x as u32, interp.uv.y as u32);
        let color: Vec3 = self.texture.get_pixel(px, py).0.into();
        let mnormal: Vec3 = self.normal_map.get_pixel(px, py).0.into();
        let normal = self.model_normal_transform.transform_vector3(mnormal);
        let intensity = normal.dot(self.light_direction).neg().max(0.0);
        debug_assert!(intensity <= 1.0 && intensity >= 0.0);
        Some(color * intensity)
    }
}

#[derive(Clone)]
struct GouradIntermediate {
    normal: Vec3,
    uv: Vec2,
    pos: Vec3,
}

impl Copy for GouradIntermediate {}

impl Mul<f32> for GouradIntermediate {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            normal: self.normal * rhs,
            uv: self.uv * rhs,
            pos: self.pos * rhs,
        }
    }
}

impl Add for GouradIntermediate {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            normal: self.normal + rhs.normal,
            uv: self.uv + rhs.uv,
            pos: self.pos + rhs.pos,
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
    let halfscreen = image_size as f32 / 2.0;

    let camera_pos = Vec3::new(0.4, 0.3, 3.0);
    let model_pos = Vec3::new(0.0, 0.0, 0.0);
    let model_scale = Vec3::splat(1.0);

    let model = Mat4::from_translation(model_pos) * Mat4::from_scale(model_scale);
    let view = Mat4::look_at_lh(camera_pos, model_pos, Vec3::Y);

    let project = Mat4::perspective_rh(PI * 0.25, 1.0, 0.001, 1.0);
    let viewport = Mat4::from_translation(Vec3::new(halfscreen, halfscreen, halfscreen))
        * Mat4::from_scale(Vec3::new(halfscreen, -halfscreen, halfscreen));

    let model_normal_transform = model.inverse().transpose();
    let model_to_screen = viewport * project * view * model;
    let light_direction = Vec3::new(0.0, -1.0, -0.5).normalize();

    let shader = Gourad::new(
        model_to_screen,
        model_normal_transform,
        texture(),
        normal_map(),
        light_direction,
    );
    shader.run(frame, &load_model())
}

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

// (position, texure coords, normal)
fn load_model() -> Vec<([(Vec3, Vec2, Vec3); 3])> {
    let obj = parse_obj(Cursor::new(include_bytes!("head.obj"))).unwrap();
    let mut ret = Vec::<[(Vec3, Vec2, Vec3); 3]>::new();
    for poly in obj.polygons {
        match poly {
            Polygon::P(_) => panic!("Model does not provide texture coords!"),
            Polygon::PT(_) => panic!("Model does not provide normals!"),
            Polygon::PN(_) => panic!("Model does not provide texture coords!"),
            Polygon::PTN(ref ptns) => {
                assert_eq!(ptns.len(), 3);
                let face = [ptns[0], ptns[1], ptns[2]];
                let face = face.map(|(p, t, n)| {
                    (
                        Vec4::from(obj.positions[p]).xyz(),
                        Vec3::from(obj.tex_coords[t]).xy(),
                        Vec3::from(obj.normals[n]).normalize(),
                    )
                });
                ret.push(face);
            }
        };
    }

    if cfg!(debug_assertions) {
        for face in &ret {
            for (_p, uv, _n) in face {
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
    let bs = include_bytes!("head_diffuse.png");
    let mut tex = image::load_from_memory_with_format(bs, ImageFormat::Png)
        .unwrap()
        .to_rgb32f();
    flip_vertical_in_place(&mut tex);
    tex
}

fn normal_map() -> image::Rgb32FImage {
    let bs = include_bytes!("head_nm.png");
    let mut tex = image::load_from_memory_with_format(bs, ImageFormat::Png)
        .unwrap()
        .to_rgb32f();
    flip_vertical_in_place(&mut tex);
    for p in tex.pixels_mut() {
        let v: Vec3 = p.0.into();
        p.0 = v.normalize().into();
    }
    tex
}
