//! Reader for Cactus `.cact` archives (format v2, tag 0x05E12A84).
//!
//! Layout (little-endian): 48 u32 + 1 f32 header, codebooks (f32), a nameless
//! directory of 44-byte records, 64-byte aligned tensor blobs, and a RAW
//! SentencePiece tokenizer dump as the last tensor. Tensors are positional, so
//! names come from `cactmap.rs` (built once by matching every record against
//! the released mirage.safetensors).
//!
//! CQ tensors are Walsh-rotated codebook quantised: each row is split in groups,
//! each group stores `bits`-wide codebook indices (LSB-first bitstream) and an
//! FP16 L2 norm; w_group = FWHT(codebook[idx] * norm) / sqrt(group).

use std::collections::HashMap;

use crate::cactmap::MAP;
use crate::{err, par, Result};

pub const TAG: u32 = 0x05e1_2a84;
const FP16: u8 = 1;
const FP32: u8 = 2;
const CQ: u8 = 3;
const REC: usize = 44;

/// IEEE half to f32.
pub fn half(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let f = (h & 0x3ff) as f32;
    match e {
        0 => s * f * (2f32).powi(-24),
        31 => {
            if f != 0.0 {
                f32::NAN
            } else {
                s * f32::INFINITY
            }
        }
        _ => s * (1.0 + f / 1024.0) * (2f32).powi(e - 15),
    }
}

/// In-place unnormalised fast Walsh-Hadamard transform (natural order).
pub fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (x, y) = (a[j], a[j + h]);
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += h << 1;
        }
        h <<= 1;
    }
}

#[derive(Debug, Clone)]
pub struct Rec {
    pub dt: u8,
    pub shape: Vec<usize>,
    pub off: usize,
    pub nb: usize,
    pub group: usize,
    pub bits: usize,
    pub name: &'static str,
    pub layer: Option<usize>,
    pub transform: &'static str,
}

impl Rec {
    fn cols(&self) -> usize {
        self.shape[1..].iter().product::<usize>().max(1)
    }
}

/// A dense row-major `[out][in]` matrix.
pub struct Mat {
    pub w: Vec<f32>,
    pub out: usize,
    pub inp: usize,
}

pub struct Cact {
    data: Vec<u8>,
    recs: Vec<Rec>,
    index: HashMap<String, usize>,
    codebooks: [Vec<f32>; 3], // 2, 3, 4 bits
}

fn key(name: &str, layer: Option<usize>) -> String {
    match layer {
        Some(l) => format!("{name}#{l}"),
        None => name.to_string(),
    }
}

impl Cact {
    pub fn new(data: Vec<u8>) -> Result<Cact> {
        let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        if data.len() < 400 || u32_at(0) != TAG {
            return err("not a Cactus v2 .cact archive");
        }
        let n = u32_at(4) as usize;
        let cbl = u32_at(8) as usize;
        if n != MAP.len() {
            return err(format!("unexpected .cact: {n} tensors, this build knows the {}-tensor Mirage layout", MAP.len()));
        }
        if cbl < 28 {
            return err("unexpected .cact codebook table");
        }
        let cb: Vec<f32> = (0..cbl).map(|i| f32::from_le_bytes(data[196 + i * 4..200 + i * 4].try_into().unwrap())).collect();
        let codebooks = [cb[0..4].to_vec(), cb[4..12].to_vec(), cb[12..28].to_vec()];
        let mut off = 196 + cbl * 4;
        let mut recs = Vec::with_capacity(n);
        let mut index = HashMap::with_capacity(n);
        for (i, &(name, layer, transform)) in MAP.iter().enumerate() {
            if off + REC > data.len() {
                return err(".cact directory truncated");
            }
            let dt = data[off];
            let nd = data[off + 1] as usize;
            let shape = (0..nd.min(4)).map(|k| u32_at(off + 4 + k * 4) as usize).collect();
            let o = u64::from_le_bytes(data[off + 20..off + 28].try_into().unwrap()) as usize;
            let nb = u64::from_le_bytes(data[off + 28..off + 36].try_into().unwrap()) as usize;
            if o.checked_add(nb).map_or(true, |e| e > data.len()) {
                return err(format!(".cact truncated at tensor {i}"));
            }
            let layer = (layer >= 0).then_some(layer as usize);
            recs.push(Rec { dt, shape, off: o, nb, group: u32_at(off + 36) as usize, bits: u32_at(off + 40) as usize, name, layer, transform });
            index.insert(key(name, layer), i);
            off += REC;
        }
        Ok(Cact { data, recs, index, codebooks })
    }

    pub fn rec(&self, name: &str, layer: Option<usize>) -> Result<&Rec> {
        match self.index.get(&key(name, layer)) {
            Some(&i) => Ok(&self.recs[i]),
            None => err(format!("missing tensor {}", key(name, layer))),
        }
    }

    fn half_at(&self, o: usize) -> f32 {
        half(u16::from_le_bytes([self.data[o], self.data[o + 1]]))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let r = self.rec(name, None)?;
        Ok(&self.data[r.off..r.off + r.nb])
    }

    fn codebook(&self, bits: usize) -> Result<&[f32]> {
        match bits {
            2..=4 => Ok(&self.codebooks[bits - 2]),
            _ => err(format!("unsupported CQ width {bits}")),
        }
    }

    /// Dequantised tensor in the archive's own layout.
    pub fn get_l(&self, name: &str, layer: Option<usize>) -> Result<Vec<f32>> {
        let r = self.rec(name, layer)?;
        let d = &self.data[r.off..r.off + r.nb];
        match r.dt {
            FP16 => Ok(d.chunks_exact(2).map(|c| half(u16::from_le_bytes([c[0], c[1]]))).collect()),
            FP32 => Ok(d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
            CQ => {
                let rows = r.shape[0];
                let cols = r.cols();
                let rr = self.rows(name, layer)?;
                let mut out = vec![0f32; rows * cols];
                par::rows(&mut out, cols, rows * cols * 16, |r0, blk| {
                    let mut tmp = vec![0f32; rr.ip];
                    for (k, row) in blk.chunks_exact_mut(cols).enumerate() {
                        rr.row_into(r0 + k, row, &mut tmp);
                    }
                });
                Ok(out)
            }
            _ => err(format!("tensor {name} is RAW")),
        }
    }

    pub fn get(&self, name: &str) -> Result<Vec<f32>> {
        self.get_l(name, None)
    }

    /// Row-wise dequantiser for a CQ tensor (for tables too large to expand).
    pub fn rows(&self, name: &str, layer: Option<usize>) -> Result<CqRows<'_>> {
        let r = self.rec(name, layer)?;
        if r.dt != CQ {
            return err(format!("{name} is not CQ"));
        }
        let cols = r.cols();
        let g = r.group;
        if g == 0 || !g.is_power_of_two() {
            return err(format!("{name}: unsupported CQ group {g}"));
        }
        let ip = cols.div_ceil(g) * g;
        let row_bytes = (ip * r.bits) >> 3;
        let ng = ip / g;
        let norm_base = r.off + r.shape[0] * row_bytes;
        if norm_base + r.shape[0] * ng * 2 > r.off + r.nb {
            return err(format!("{name}: unexpected CQ layout"));
        }
        Ok(CqRows { cact: self, off: r.off, cols, g, bits: r.bits, ip, row_bytes, ng, norm_base, cb: self.codebook(r.bits)?, inv: 1.0 / (g as f32).sqrt() })
    }

    /// Dense weight as row-major `[out][in]`.
    pub fn dense_l(&self, name: &str, layer: Option<usize>) -> Result<Mat> {
        let r = self.rec(name, layer)?.clone();
        let w = self.get_l(name, layer)?;
        if r.transform == "T" {
            return Ok(Mat { w, out: r.shape[0], inp: r.cols() });
        }
        if r.shape.len() != 2 {
            return err(format!("{name}: not a matrix"));
        }
        let (i_n, o_n) = (r.shape[0], r.shape[1]);
        let mut t = vec![0f32; i_n * o_n];
        for i in 0..i_n {
            for o in 0..o_n {
                t[o * i_n + i] = w[i * o_n + o];
            }
        }
        Ok(Mat { w: t, out: o_n, inp: i_n })
    }

    pub fn dense(&self, name: &str) -> Result<Mat> {
        self.dense_l(name, None)
    }
}

pub struct CqRows<'a> {
    cact: &'a Cact,
    off: usize,
    pub cols: usize,
    g: usize,
    bits: usize,
    pub ip: usize,
    row_bytes: usize,
    ng: usize,
    norm_base: usize,
    cb: &'a [f32],
    inv: f32,
}

impl CqRows<'_> {
    /// Dequantise row `row` into `out[..cols]`; `tmp` has `ip` values.
    pub fn row_into(&self, row: usize, out: &mut [f32], tmp: &mut [f32]) {
        let u8s = &self.cact.data;
        let base = self.off + row * self.row_bytes;
        let mask = (1usize << self.bits) - 1;
        for (k, t) in tmp.iter_mut().enumerate().take(self.ip) {
            let bit = k * self.bits;
            let byte = base + (bit >> 3);
            let sh = bit & 7;
            let mut v = u8s[byte] as usize;
            if sh + self.bits > 8 {
                v |= (u8s[byte + 1] as usize) << 8;
            }
            *t = self.cb[(v >> sh) & mask];
        }
        for gi in 0..self.ng {
            let nrm = self.cact.half_at(self.norm_base + (row * self.ng + gi) * 2) * self.inv;
            let s = &mut tmp[gi * self.g..(gi + 1) * self.g];
            for v in s.iter_mut() {
                *v *= nrm;
            }
            fwht(s);
        }
        out[..self.cols].copy_from_slice(&tmp[..self.cols]);
    }
}
