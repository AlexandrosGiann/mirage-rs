//! Convolutional CTC text reader: crops a text box from the original image,
//! scales it to 48 px high (aspect preserved) and reads it in overlapping
//! 320 px windows; greedy CTC collapse across the joined window centres.

use crate::cact::{Cact, Mat};
use crate::image::{resize_area, Rgb};
use crate::ops::*;
use crate::Result;

const HEIGHT: usize = 48;
const WIN: usize = 320;
const OVERLAP: usize = 64;
const DIM: usize = 384;
const BLANK: usize = 0;

struct Block {
    norm: Vec<f32>,
    mix: Vec<f32>,
    up: Mat,
    down: Mat,
}

pub struct Reader {
    w0: Vec<f32>,
    stage_norm: Vec<Vec<f32>>,
    dw: Vec<Vec<f32>>,
    pw: Vec<Mat>,
    blocks: Vec<Block>,
    fin: Vec<f32>,
    out: Mat,
    bias: Vec<f32>,
    classes: usize,
    charset: Vec<char>,
}

impl Reader {
    pub fn new(c: &Cact) -> Result<Self> {
        let g = |n: &str| c.get(&format!("reader/{n}"));
        let dn = |n: &str| c.dense(&format!("reader/{n}"));
        let bias = g("out/bias")?;
        Ok(Reader {
            w0: g("w0")?,
            stage_norm: (0..4).map(|i| g(&format!("stage_norm_{i}/scale"))).collect::<Result<_>>()?,
            dw: (0..3).map(|i| g(&format!("dw_{i}"))).collect::<Result<_>>()?,
            pw: (0..3).map(|i| dn(&format!("pw_{i}/kernel"))).collect::<Result<_>>()?,
            blocks: (0..4)
                .map(|i| -> Result<Block> {
                    Ok(Block { norm: g(&format!("norm_{i}/scale"))?, mix: g(&format!("mix_{i}"))?, up: dn(&format!("up_{i}/kernel"))?, down: dn(&format!("down_{i}/kernel"))? })
                })
                .collect::<Result<_>>()?,
            fin: g("final_norm/scale")?,
            out: dn("out/kernel")?,
            classes: bias.len(),
            bias,
            charset: c.get("reader_charset")?.iter().map(|&v| char::from_u32(v as u32).unwrap_or('\u{fffd}')).collect(),
        })
    }

    /// Per-frame class logits for one normalised window `[48][w][3]`; returns (logits, frames).
    fn frames(&self, j: &[f32], w: usize) -> (Vec<f32>, usize) {
        let (mut g, mut h, mut ww) = conv3(j, HEIGHT, w, 3, &self.w0, 32, 2, 2, Some(1));
        rmsnorm_in_place(&mut g, 32, Some(&self.stage_norm[0]));
        let strides = [(2, 2), (2, 1), (2, 1)];
        let mut c = 32;
        for (s, &(sh, sw)) in strides.iter().enumerate() {
            let mut a = g.clone();
            silu_in_place(&mut a);
            let (d, h2, w2) = dwconv3(&a, h, ww, c, &self.dw[s], sh, sw, 1);
            h = h2;
            ww = w2;
            let pw = &self.pw[s];
            g = linear(&d, h * ww, &pw.w, pw.out, c, None);
            c = pw.out;
            rmsnorm_in_place(&mut g, c, Some(&self.stage_norm[s + 1]));
        }
        // mean over height of SiLU(G4)
        let t = ww;
        let mut z = vec![0f32; t * DIM];
        for y in 0..h {
            for x in 0..ww {
                for k in 0..DIM {
                    z[x * DIM + k] += silu(g[(y * ww + x) * DIM + k]) / h as f32;
                }
            }
        }
        let mix_pad = same_pad(t, 1, 5);
        for b in &self.blocks {
            let n = rmsnorm(&z, DIM, Some(&b.norm));
            let (hm, _, _) = dwconv(&n, 1, t, DIM, &b.mix, 1, 5, 1, 1, 0, mix_pad);
            let ug = linear(&hm, t, &b.up.w, 2 * DIM, DIM, None);
            let mut m = vec![0f32; t * DIM];
            for tt in 0..t {
                for k in 0..DIM {
                    let v = ug[tt * 2 * DIM + k];
                    let gg = ug[tt * 2 * DIM + DIM + k];
                    m[tt * DIM + k] = v * silu(gg);
                }
            }
            let o = linear(&m, t, &b.down.w, DIM, DIM, None);
            add_in_place(&mut z, &o);
        }
        let zn = rmsnorm(&z, DIM, Some(&self.fin));
        (linear(&zn, t, &self.out.w, self.classes, DIM, Some(&self.bias)), t)
    }

    /// Read the text in `bx` = [x0,y0,x1,y1] (original pixels). Returns (text, confidence).
    pub fn read(&self, img: &Rgb, bx: [f32; 4]) -> (String, f32) {
        let [x0, y0, x1, y1] = bx;
        let bw = (x1 - x0).max(1.0);
        let bh = (y1 - y0).max(1.0);
        let w = ((HEIGHT as f32 * bw / bh).round() as usize).max(4);
        let cx0 = (x0.floor().max(0.0) as usize).min(img.width - 1);
        let cy0 = (y0.floor().max(0.0) as usize).min(img.height - 1);
        let cx1 = img.width.min((cx0 + 1).max(x1.round().max(0.0) as usize));
        let cy1 = img.height.min((cy0 + 1).max(y1.round().max(0.0) as usize));
        let (cw, ch) = (cx1 - cx0, cy1 - cy0);
        let mut sub = vec![0u8; cw * ch * 3];
        for y in 0..ch {
            let s = ((cy0 + y) * img.width + cx0) * 3;
            sub[y * cw * 3..(y + 1) * cw * 3].copy_from_slice(&img.data[s..s + cw * 3]);
        }
        let px = resize_area(&Rgb { width: cw, height: ch, data: sub }, w, HEIGHT);
        let norm: Vec<f32> = px.iter().map(|&v| v / 127.5 - 1.0).collect();
        let step = WIN - OVERLAP;
        let mut starts = Vec::new();
        if w <= WIN {
            starts.push(0);
        } else {
            let mut s = 0;
            while s + WIN < w {
                starts.push(s);
                s += step;
            }
            starts.push(w - WIN);
        }
        let fw = 4.0f64; // frames are 4 px wide
        let mut covered = 0f64;
        let mut frames = Vec::new();
        let (mut score, mut n_score) = (0f64, 0usize);
        for (si, &s) in starts.iter().enumerate() {
            let ww = WIN.min(w - s);
            let mut win = vec![-1f32; HEIGHT * WIN * 3];
            for y in 0..HEIGHT {
                win[y * WIN * 3..y * WIN * 3 + ww * 3].copy_from_slice(&norm[(y * w + s) * 3..(y * w + s + ww) * 3]);
            }
            let (logits, _) = self.frames(&win, WIN);
            let t_n = ww.div_ceil(4);
            let g0 = s as f64 / fw;
            // keep frames from the window centre: up to the midpoint of the overlap with the next window
            let end = if si + 1 < starts.len() { ((starts[si + 1] as f64 / fw + g0 + t_n as f64) / 2.0).round() } else { g0 + t_n as f64 };
            for t in 0..t_n {
                let gt = g0 + t as f64;
                if gt < covered || gt >= end {
                    continue;
                }
                let l = &logits[t * self.classes..(t + 1) * self.classes];
                let (mut m, mut am) = (f32::NEG_INFINITY, 0);
                for (k, &v) in l.iter().enumerate() {
                    if v > m {
                        m = v;
                        am = k;
                    }
                }
                let z: f64 = l.iter().map(|&v| ((v - m) as f64).exp()).sum();
                frames.push(am);
                if am != BLANK {
                    score += 1.0 / z;
                    n_score += 1;
                }
            }
            covered = end;
        }
        let mut out = String::new();
        let mut prev = usize::MAX;
        for &k in &frames {
            if k != prev && k != BLANK {
                if let Some(&ch) = self.charset.get(k - 1) {
                    out.push(ch);
                }
            }
            prev = k;
        }
        (out, if n_score > 0 { (score / n_score as f64) as f32 } else { 0.0 })
    }
}
