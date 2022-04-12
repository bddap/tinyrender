use std::{fs::File, io::BufWriter, path::PathBuf};

use common::to_pixel;
use image::{imageops::flip_vertical_in_place, ImageBuffer, ImageOutputFormat, Rgba, RgbaImage};

mod common;
mod consts;
mod lesson1;
mod lesson2;
mod lesson3;
mod lesson3_5;
mod lesson3_6;
mod lesson3_7;

use consts::*;
use structopt::StructOpt;

pub const RENDERS: &[for<'r> fn(&'r mut ImageBuffer<Rgba<u8>, Vec<u8>>)] = &[
    lesson1::render,
    lesson2::render1,
    lesson2::render2,
    lesson3::triangles,
    lesson3_5::triangles,
    lesson3_6::triangles,
    lesson3_6::triangles2,
    lesson3_7::render,
    lesson3_7::render,
];

#[derive(structopt::StructOpt)]
struct Args {
    /// output path, will be written as png
    #[structopt(short, long)]
    output: PathBuf,

    #[structopt(short, long, default_value = "2160")]
    size: u32,

    /// which render to run, default latest
    #[structopt(short, long)]
    render: Option<usize>,
}

fn main() {
    let args = Args::from_args();
    let lesson = RENDERS
        .get(args.render.unwrap_or(RENDERS.len()).saturating_sub(1))
        .unwrap();
    let mut outfile = BufWriter::new(File::create(args.output).unwrap());
    let mut img = RgbaImage::from_pixel(args.size, args.size, to_pixel(black()));

    (lesson)(&mut img);
    flip_vertical_in_place(&mut img);

    img.write_to(&mut outfile, ImageOutputFormat::Png).unwrap();
}
