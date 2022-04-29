// this one needs --release mode

// There are fireflies on the vertices!

use std::{
    cell::RefCell,
    f32::consts::PI,
    io::Cursor,
    ops::{Add, DivAssign, Mul},
};

use crate::common::to_pixel;

use glam::{Mat4, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use image::{imageops::flip_vertical_in_place, ImageFormat, RgbaImage};
use obj::raw::{object::Polygon, parse_obj};
use rand::{prelude::SmallRng, SeedableRng};
use tap::Conv;

struct Main {
    model_to_screen: Mat4,
    texture: image::Rgb32FImage,
    texsize: Vec2,
    light_direction: Vec3,
    bake_out: Frame,
}

impl Shader for Main {
    type Varying = MainVarying;
    type Intermediate = Nothing;

    fn vertex(&self, vert: Vec3, _: Vec3, uv: Vec2) -> (Vec3, Self::Varying, Self::Intermediate) {
        let pos = self.model_to_screen.project_point3(vert);
        (pos, MainVarying { uv, pos }, Nothing)
    }

    fn fragment(&self, interp: Self::Varying, _: [Self::Intermediate; 3]) -> Option<Vec3> {
        let uvt = interp.uv * self.texsize;
        let (px, py) = (uvt.x as u32, uvt.y as u32);
        let color: Vec3 = self.texture.get_pixel(px, py).0.conv::<Vec3>();

        let uvt = interp.uv * self.bake_out.size();
        let (px, py) = (uvt.x as u32, uvt.y as u32);
        let ambient = self.bake_out.try_get(px, py).unwrap().x;

        let brightness = 0.3 + ambient * 0.7;

        Some((color * brightness).min(Vec3::splat(1.0)))
        // Some(Vec3::splat(brightness))
    }
}

struct Prebake {
    model_to_screen: Mat4,
}

impl Shader for Prebake {
    type Varying = Vec3;
    type Intermediate = Nothing;

    fn vertex(&self, vert: Vec3, _: Vec3, _: Vec2) -> (Vec3, Self::Varying, Self::Intermediate) {
        let pos = self.model_to_screen.project_point3(vert);
        (pos, pos, Nothing)
    }

    fn fragment(&self, interp: Self::Varying, _: [Self::Intermediate; 3]) -> Option<Vec3> {
        Some(interp)
    }
}

struct PrebakePass2<'a> {
    model_to_screen: Mat4,
    pass1: &'a Frame,
    tex: RefCell<Frame>,
}

impl<'a> Shader for PrebakePass2<'a> {
    type Varying = PrebakeVarying;
    type Intermediate = Nothing;

    fn vertex(&self, vert: Vec3, _: Vec3, uv: Vec2) -> (Vec3, Self::Varying, Self::Intermediate) {
        let pos = self.model_to_screen.project_point3(vert);
        (pos, PrebakeVarying { pos, uv }, Nothing)
    }

    fn fragment(&self, interp: Self::Varying, _: [Self::Intermediate; 3]) -> Option<Vec3> {
        let (px, py) = (interp.pos.x as u32, interp.pos.y as u32);
        let from_pass1 = self.pass1.try_get(px, py).unwrap();

        if (from_pass1.z - interp.pos.z).abs() < 0.001 {
            let mut tex = self.tex.borrow_mut();
            let uvt = interp.uv * tex.size();
            let (px, py) = (uvt.x as u32, uvt.y as u32);
            *tex.get_mut(px, py) = Vec4::splat(1.0);
        }

        None
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
    let obj = load_model();
    let texture = texture();
    let texsize = Vec2::new(texture.width() as f32, texture.height() as f32);
    let halfscreen = image_size as f32 / 2.0;

    let mut rng = SmallRng::seed_from_u64(0);

    let s = 1;
    let mut pass1 = Frame::new(frame.width() * s, frame.height() * s);
    let mut bakestep2_dump = Frame::new(frame.width() * s, frame.height() * s);

    let bake_steps = 1000;
    let mut bake_out = Frame::new(frame.width(), frame.height());
    for _ in 0..bake_steps {
        let camera_pos = random_point_on_sphere(&mut rng) * 2.0;
        let model_pos = Vec3::new(0.0, 0.0, 0.0);
        let model_scale = Vec3::splat(1.0);
        let model = Mat4::from_translation(model_pos)
            * Mat4::from_scale(model_scale)
            * Mat4::from_axis_angle(Vec3::X, 0.2);
        let view = Mat4::look_at_lh(camera_pos, model_pos, Vec3::Y);
        let project = Mat4::perspective_rh(PI * 0.25, 1.0, 0.001, 1.0);
        let viewport = Mat4::from_translation(Vec3::new(halfscreen, halfscreen, 1.0))
            * Mat4::from_scale(Vec3::new(halfscreen, -halfscreen, 1.0));
        let model_to_screen = viewport * project * view * model;

        // using screen size for the light map so I can do visualizations later
        // maybe an animation of the light map being populated
        // script:
        //   show light map being populated
        //   pop model onto screen without color
        //   add color to model
        // alternate script:
        //   show everything
        //   populate lightmap step by step while rotating model
        let tex = Frame::new(frame.width(), frame.height());

        let bake1 = Prebake { model_to_screen };

        bake1.run(&mut pass1, &obj);
        let bake2 = PrebakePass2 {
            model_to_screen,
            pass1: &pass1,
            tex: RefCell::new(tex),
        };
        bake2.run(&mut bakestep2_dump, &obj);
        bake_out.add_other(&bake2.tex.into_inner());
    }
    bake_out /= bake_steps as f32;
    frame.add_other(&bake_out);

    let camera_pos = Vec3::new(1.0, 0.0, 3.0);
    let model_pos = Vec3::new(0.0, 0.0, 0.0);
    let model_scale = Vec3::splat(1.0);

    let model = Mat4::from_translation(model_pos)
        * Mat4::from_scale(model_scale)
        * Mat4::from_axis_angle(Vec3::X, 0.2);
    let view = Mat4::look_at_lh(camera_pos, model_pos, Vec3::Y);

    let project = Mat4::perspective_rh(PI * 0.25, 1.0, 0.001, 1.0);
    let viewport = Mat4::from_translation(Vec3::new(halfscreen, halfscreen, 1.0))
        * Mat4::from_scale(Vec3::new(halfscreen, -halfscreen, 1.0));

    let model_to_screen = viewport * project * view * model;

    let light_direction = Vec3::new(-1.0, -1.0, -1.0).normalize();

    let shader = Main {
        model_to_screen,
        texture,
        light_direction,
        texsize,
        bake_out,
    };
    shader.assert_valid();
    shader.run(frame, &obj)
}

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
        debug_assert!(y < self.height());
        self.pix.get_mut(x + y * self.width).unwrap()
    }

    fn try_get(&self, x: u32, y: u32) -> Option<&Vec4> {
        let (x, y) = (x as usize, y as usize);
        if x >= self.width || y >= self.height() {
            None
        } else {
            Some(&self.pix[x + y * self.width])
        }
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

    fn width(&self) -> usize {
        self.width
    }

    fn size(&self) -> Vec2 {
        Vec2::new(self.width() as f32, self.height() as f32)
    }

    fn add_other(&mut self, rhs: &Self) {
        assert_eq!(self.height(), rhs.height());
        assert_eq!(self.width(), rhs.width());
        for (sp, rhs) in self.pix.iter_mut().zip(&rhs.pix) {
            *sp += *rhs;
        }
    }
}

impl DivAssign<f32> for Frame {
    fn div_assign(&mut self, rhs: f32) {
        for sp in self.pix.iter_mut() {
            *sp /= rhs;
        }
    }
}

trait Shader {
    type Varying: Mul<f32, Output = Self::Varying>
        + Add<Self::Varying, Output = Self::Varying>
        + Copy;
    type Intermediate: Copy;

    fn vertex(
        &self,
        vert: Vec3,
        normal: Vec3,
        uv: Vec2,
    ) -> (Vec3, Self::Varying, Self::Intermediate);

    fn fragment(&self, interp: Self::Varying, interm: [Self::Intermediate; 3]) -> Option<Vec3>;

    fn run(&self, target: &mut Frame, model: &[[(Vec3, Vec2, Vec3); 3]]) {
        let screen_size = UVec2::new(target.width as u32, target.height() as u32);

        for tri in model {
            let vertout = tri.map(|(pos, uv, normal)| self.vertex(pos, normal, uv));
            let verts = vertout.map(|(v, _, _)| v);
            // todo: experiment with animations that mess with backface culling
            if normal(verts).z <= 0.0 {
                let zs = verts.map(|pos| pos.z);
                let verts = verts.map(|pos| pos.xy().as_uvec2());
                let varyings = vertout.map(|(_, var, _)| var);
                let intermediates = vertout.map(|(_, _, int)| int);

                for_coord_in_triangle(screen_size, verts, |pos, bar| {
                    let z = bary_interp(bar, zs);
                    let pix = target.get_mut(pos.x, pos.y);
                    if pix.w >= z {
                        return;
                    }
                    let interp = bary_interp(bar, varyings.clone());
                    if let Some(color) = self.fragment(interp, intermediates) {
                        *pix = color.extend(z);
                    }
                });
            }
        }
    }
}

impl Main {
    fn assert_valid(&self) {
        assert!(is_normalized(self.light_direction));
        assert_eq!(
            self.texsize,
            Vec2::new(self.texture.width() as f32, self.texture.height() as f32)
        );
    }
}

#[derive(Clone)]
struct MainVarying {
    uv: Vec2,
    pos: Vec3,
}

impl Copy for MainVarying {}

impl Mul<f32> for MainVarying {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            uv: self.uv * rhs,
            pos: self.pos * rhs,
        }
    }
}

impl Add for MainVarying {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            uv: self.uv + rhs.uv,
            pos: self.pos + rhs.pos,
        }
    }
}

#[derive(Clone)]
struct PrebakeVarying {
    uv: Vec2,
    pos: Vec3,
}

impl Copy for PrebakeVarying {}

impl Mul<f32> for PrebakeVarying {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            uv: self.uv * rhs,
            pos: self.pos * rhs,
        }
    }
}

impl Add for PrebakeVarying {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            uv: self.uv + rhs.uv,
            pos: self.pos + rhs.pos,
        }
    }
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

fn is_normalized(a: Vec3) -> bool {
    (a.length() - 1.0).abs() < 0.0001
}

fn normal(triangle: [Vec3; 3]) -> Vec3 {
    (triangle[2] - triangle[0])
        .cross(triangle[1] - triangle[0])
        .normalize()
}

// (position, texure coords, normal)
fn load_model() -> Vec<([(Vec3, Vec2, Vec3); 3])> {
    let obj = parse_obj(Cursor::new(include_bytes!(
        "./obj/diablo3_pose/diablo3_pose.obj"
    )))
    .unwrap();
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
    let bs = include_bytes!("./obj/diablo3_pose/diablo3_pose_diffuse.png");
    let mut tex = image::load_from_memory_with_format(bs, ImageFormat::Png)
        .unwrap()
        .to_rgb32f();
    flip_vertical_in_place(&mut tex);
    tex
}

#[derive(Clone)]
struct Nothing;

impl Copy for Nothing {}

impl Mul<f32> for Nothing {
    type Output = Self;

    fn mul(self, _: f32) -> Self::Output {
        Self
    }
}

impl Add for Nothing {
    type Output = Self;

    fn add(self, _: Self) -> Self {
        Self
    }
}

fn random_point_on_sphere(rng: &mut impl rand::Rng) -> Vec3 {
    let [a, b, c, d]: [f32; 4] = rng.gen();
    let ss = a * a + b * b + c * c + d * d;

    // https://mathworld.wolfram.com/SpherePointPicking.html
    let x = 2.0 * (b * d + a * c) / ss;
    let y = 2.0 * (c * d - a * b) / ss;
    let z = (a * a + d * d - b * b - c * c) / ss;

    Vec3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::random_point_on_sphere;

    #[test]
    fn on_shere() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        for _ in 0..10000 {
            assert!((random_point_on_sphere(&mut rng).length() - 1.0).abs() < 0.00001);
        }
    }
}
