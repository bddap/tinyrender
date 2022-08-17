#[derive(serde::Serialize, serde::Deserialize)]
pub struct Atom {
    pub element: Element,
    pub pos: [f32; 3],
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum Element {
    H,
    C,
    O,
    N,
    P,
    S,
    Na,
    Mg,
    Cl,
    K,
    Ca,
    Fe,
    Mn,
    Co,
    Cr,
    I,
    Zn,
    Cu,
    F,
    Al,
    Se,
    V,
    Unknown,
}
