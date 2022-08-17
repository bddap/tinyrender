//! Some pbd files are large enough that attempting to parse them comsumes all available memory.
//! Luckily we only care about elements and positions here.
//! This script reads the interesting values from a pdb file and serializes them in a compact manner
//! using bincode.
//! input is accepted on stdin and output is printed to stdout
//!
//! ```bash
//! cat tmp/2btv.cif | cargo run --release --bin convert-cif | gzip -9 > src/lesson9c/2btv.bin.gz
//! ```

use std::io::{BufRead, BufReader, BufWriter};

use tinyrender::{Atom, Element};

fn main() -> anyhow::Result<()> {
    let mut stdout = BufWriter::with_capacity(128 * 1024, std::io::stdout());
    let mut stdin = BufReader::new(std::io::stdin());
    let atoms: Vec<Atom> = parse_pdb_jankily_yolo(&mut stdin).collect::<anyhow::Result<_>>()?;
    bincode::serialize_into(&mut stdout, &atoms)?;
    Ok(())
}

fn parse_pdb_jankily_yolo(
    read: &mut impl BufRead,
) -> impl Iterator<Item = anyhow::Result<Atom>> + '_ {
    read.lines()
        .map(|r: std::io::Result<String>| {
            let ln = r?;
            if !ln.starts_with("ATOM ") {
                return Ok(None);
            }
            let mut sp = ln.split_whitespace();

            let _ = sp.next(); // ATOM
            let _ = sp.next(); // 1
            let element = sp.next(); // N
            let _ = sp.next(); // N
            let _ = sp.next(); // .
            let _ = sp.next(); // VAL
            let _ = sp.next(); // A
            let _ = sp.next(); // 1
            let _ = sp.next(); // 57
            let _ = sp.next(); // ?
            let x = sp.next(); // 216.934
            let y = sp.next(); // -8.171
            let z = sp.next(); // 276.852
            let _ = sp.next(); // 1.00
            let _ = sp.next(); // 21.19
            let _ = sp.next(); // ?
            let _ = sp.next(); // ?
            let _ = sp.next(); // ?
            let _ = sp.next(); // ?
            let _ = sp.next(); // ?
            let _ = sp.next(); // ?
            let _ = sp.next(); // 57
            let _ = sp.next(); // VAL
            let _ = sp.next(); // A
            let _ = sp.next(); // N
            let _ = sp.next(); // 1
            let stop = sp.next();

            match (element, x, y, z, stop) {
                (Some(element), Some(x), Some(y), Some(z), None) => {
                    let pos: [f32; 3] = [x.parse()?, y.parse()?, z.parse()?];
                    let element = str_to_elem(&element);
                    Ok(Some(Atom { element, pos }))
                }
                _ => Err(anyhow::anyhow!("wrong number of rows")),
            }
        })
        .filter_map(|r| r.transpose())
}

fn str_to_elem(s: &str) -> Element {
    match s {
        "H" => Element::H,
        "C" => Element::C,
        "O" => Element::O,
        "N" => Element::N,
        "P" => Element::P,
        "S" => Element::S,
        "Na" => Element::Na,
        "Mg" => Element::Mg,
        "Cl" => Element::Cl,
        "K" => Element::K,
        "Ca" => Element::Ca,
        "Fe" => Element::Fe,
        "Mn" => Element::Mn,
        "Co" => Element::Co,
        "Cr" => Element::Cr,
        "I" => Element::I,
        "Zn" => Element::Zn,
        "Cu" => Element::Cu,
        "F" => Element::F,
        "Al" => Element::Al,
        "Se" => Element::Se,
        "V" => Element::V,
        _ => Element::Unknown,
    }
}
