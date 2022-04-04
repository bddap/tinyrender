use crate::{common::to_pixel, consts::red};

use image::RgbaImage;

pub fn render(image: &mut RgbaImage) {
    let (w, h) = image.dimensions();
    assert!(w > 0);
    assert!(h > 0);

    image.put_pixel(w / 2, h / 2, to_pixel(red()));
}
