use std::{fs::File, io::BufWriter, path::PathBuf};

use common::to_pixel;
use image::{imageops::flip_vertical_in_place, ImageOutputFormat, RgbaImage};

mod common;
mod consts;
mod lesson1;
mod lesson2;

use consts::*;
use structopt::StructOpt;

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
