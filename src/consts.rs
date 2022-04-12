use glam::Vec3;

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

pub const HEAD_OBJ_BYTES: &[u8] = include_bytes!("head.obj");
