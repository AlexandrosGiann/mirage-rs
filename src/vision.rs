//! Mirage image encoder, feature-pyramid neck, proposal / mask heads and the
//! box-reference slots, following the published forward pass.

use crate::cact::{fwht, Cact, Mat};
use crate::ops::*;
use crate::{par, Result};

pub const CANVAS: usize = 512;
const D: usize = 512;

/// sequency order: u_j = FWHT(x)[s(j)] / sqrt(n), s(j) = rev(j ^ j>>1)
fn seq() -> Vec<usize> {
    let q = 9;
    (0..1usize << q)
        .map(|j| {
            let g = j ^ (j >> 1);
            let mut r = 0;
            for b in 0..q {
                r |= ((g >> b) & 1) << (q - 1 - b);
            }
            r
        })
        .collect()
}

struct HConvGlu {
    norm: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
}

/// (Bu)_j = sum_t w[j][t+32] u[j+t]
fn band(w: &[f32], u: &[f32], out: &mut [f32]) {
    for j in 0..D {
        let k0 = j.saturating_sub(32);
        let k1 = (j + 32).min(D - 1);
        let wo = j * 64 + 32; // w[j*65 + (k - j) + 32] = w[wo + k]
        out[j] = dot(&w[wo + k0..wo + k1 + 1], &u[k0..=k1]);
    }
}

impl HConvGlu {
    fn new(c: &Cact, p: &str) -> Result<Self> {
        Ok(HConvGlu {
            norm: c.get(&format!("{p}hconv_norm/scale"))?,
            g: c.get(&format!("{p}hconv/gate_conv"))?,
            b: c.get(&format!("{p}hconv/gate_bias"))?,
            v: c.get(&format!("{p}hconv/value_conv"))?,
            o: c.get(&format!("{p}hconv/out_conv"))?,
        })
    }

    /// H = W B_o (GELU(B_g u + b) * B_v u),  u = W N(x)
    fn run(&self, x: &[f32], t: usize, seq: &[usize]) -> Vec<f32> {
        let xn = rmsnorm(x, D, Some(&self.norm));
        let mut h = vec![0f32; t * D];
        let inv = 1.0 / (D as f32).sqrt();
        let walsh = |src: &[f32], dst: &mut [f32], tmp: &mut [f32]| {
            tmp.copy_from_slice(src);
            fwht(tmp);
            for j in 0..D {
                dst[j] = tmp[seq[j]] * inv;
            }
        };
        par::rows(&mut h, D, t * D * 65 * 3, |t0, blk| {
            let (mut u, mut a, mut v, mut tmp) = (vec![0f32; D], vec![0f32; D], vec![0f32; D], vec![0f32; D]);
            for (k, hr) in blk.chunks_exact_mut(D).enumerate() {
                let p = t0 + k;
                walsh(&xn[p * D..(p + 1) * D], &mut u, &mut tmp);
                band(&self.g, &u, &mut a);
                band(&self.v, &u, &mut v);
                for j in 0..D {
                    a[j] = gelu(a[j] + self.b[j]) * v[j];
                }
                band(&self.o, &a, &mut u);
                walsh(&u, hr, &mut tmp);
            }
        });
        h
    }
}

fn rope2d(x: &mut [f32], row: usize, col: usize) {
    for r in 0..16 {
        let pos = if r < 8 { row } else { col } as f64;
        let w = pos * 1e4f64.powf(-2.0 * (r & 7) as f64 / 16.0);
        let (c, s) = (w.cos() as f32, w.sin() as f32);
        let (u, v) = (x[r], x[r + 16]);
        x[r] = u * c - v * s;
        x[r + 16] = v * c + u * s;
    }
}

struct Body {
    mix_norm: Vec<f32>,
    dwv: Vec<f32>,
    bv: Vec<f32>,
    dwg: Vec<f32>,
    bg: Vec<f32>,
    dwo: Vec<f32>,
    hconv: HConvGlu,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Top {
    norm: Vec<f32>,
    q: Mat,
    k: Mat,
    v: Mat,
    gate: Mat,
    out: Mat,
    qn: Vec<f32>,
    kn: Vec<f32>,
    post: Vec<f32>,
    gate_a: f32,
    hconv: HConvGlu,
}

pub struct Res {
    norm: Vec<f32>,
    dw: Vec<f32>,
    pw: Mat,
}

struct MaskHead {
    blocks: Vec<Res>,
    fin: Vec<f32>,
    proj: Mat,
}

/// Encoder output.
pub struct Encoded {
    /// image memory, 16x16x512
    pub e: Vec<f32>,
    s4: Vec<f32>,
    s8: Vec<f32>,
    s16: Vec<f32>,
    pub gh: usize,
    pub gw: usize,
    pub vh: usize,
    pub vw: usize,
}

pub struct VisionModel {
    w0: Vec<f32>,
    stem_norms: Vec<Vec<f32>>,
    stem_dw: Vec<Vec<f32>>,
    stem_pw: Vec<Mat>,
    body: Vec<Body>,
    top: Vec<Top>,
    registers: Vec<f32>,
    enc_norm: Vec<f32>,
    level_norms: Vec<Vec<f32>>,
    level_gates: Vec<f32>,
    coarse: Mat,
    mid: Mat,
    fine: Mat,
    edge: Mat,
    neck: Res,
    blocks: Vec<Res>,
    fin: Vec<f32>,
    out: Mat,
    bias: Vec<f32>,
    proto: MaskHead,
    coef: MaskHead,
    slot_proj: Mat,
    slot_pool: Mat,
    slot_norm: Vec<f32>,
    seq: Vec<usize>,
}

impl VisionModel {
    pub fn new(c: &Cact) -> Result<Self> {
        let g = |n: &str| c.get(n);
        let dn = |n: &str| c.dense(&format!("dense/{n}"));
        let res = |n: &str, d: &str, p: &str| -> Result<Res> { Ok(Res { norm: g(n)?, dw: g(d)?, pw: c.dense(p)? }) };
        let mut body = Vec::new();
        for l in 0..16 {
            let p = format!("body_{l}/");
            body.push(Body {
                mix_norm: g(&format!("{p}mix_norm/scale"))?,
                dwv: g(&format!("{p}mix/dw_value"))?,
                bv: g(&format!("{p}mix/b_value"))?,
                dwg: g(&format!("{p}mix/dw_gate"))?,
                bg: g(&format!("{p}mix/b_gate"))?,
                dwo: g(&format!("{p}mix/dw_down"))?,
                hconv: HConvGlu::new(c, &p)?,
                gamma: g(&format!("{p}grn/gamma"))?,
                beta: g(&format!("{p}grn/beta"))?,
            });
        }
        let mut top = Vec::new();
        for l in 0..12 {
            let p = format!("top_{l}/");
            let d = |n: &str| c.dense(&format!("{p}self_attn/{n}/kernel"));
            top.push(Top {
                norm: g(&format!("{p}attn_norm/scale"))?,
                q: d("q_proj")?,
                k: d("k_proj")?,
                v: d("v_proj")?,
                gate: d("gate_proj")?,
                out: d("out_proj")?,
                qn: g(&format!("{p}self_attn/q_norm/scale"))?,
                kn: g(&format!("{p}self_attn/k_norm/scale"))?,
                post: g(&format!("{p}post_attn_norm/scale"))?,
                gate_a: sigmoid(g(&format!("{p}attn_gate"))?[0]),
                hconv: HConvGlu::new(c, &p)?,
            });
        }
        let head = |n: &str, proj: &str| -> Result<MaskHead> {
            Ok(MaskHead {
                blocks: (0..2)
                    .map(|i| res(&format!("dense/{n}_norm_{i}/scale"), &format!("dense/{n}_dw_{i}"), &format!("dense/{n}_pw_{i}/kernel")))
                    .collect::<Result<_>>()?,
                fin: g(&format!("dense/{n}_final_norm/scale"))?,
                proj: dn(&format!("{proj}/kernel"))?,
            })
        };
        Ok(VisionModel {
            w0: g("stem/w0")?,
            stem_norms: (0..5).map(|i| g(&format!("stem/norm_{i}/scale"))).collect::<Result<_>>()?,
            stem_dw: (1..5).map(|i| g(&format!("stem/dw_{i}"))).collect::<Result<_>>()?,
            stem_pw: (1..5).map(|i| c.dense(&format!("stem/pw_{i}/kernel"))).collect::<Result<_>>()?,
            body,
            top,
            registers: g("registers")?,
            enc_norm: g("enc_norm/scale")?,
            level_norms: (0..12).map(|i| g(&format!("level_norm_{i}/scale"))).collect::<Result<_>>()?,
            level_gates: g("level_gates")?,
            coarse: dn("coarse/kernel")?,
            mid: dn("mid/kernel")?,
            fine: dn("fine/kernel")?,
            edge: dn("edge/kernel")?,
            neck: res("dense/neck_norm/scale", "dense/neck_dw", "dense/neck_pw/kernel")?,
            blocks: (0..3)
                .map(|i| res(&format!("dense/block_norm_{i}/scale"), &format!("dense/dw_{i}"), &format!("dense/pw_{i}/kernel")))
                .collect::<Result<_>>()?,
            fin: g("dense/final_norm/scale")?,
            out: dn("out/kernel")?,
            bias: g("dense/bias")?,
            proto: head("proto", "protos")?,
            coef: head("coef", "coef")?,
            slot_proj: c.dense("slot_proj/kernel")?,
            slot_pool: c.dense("slot_pool/kernel")?,
            slot_norm: g("slot_norm/scale")?,
            seq: seq(),
        })
    }

    /// `img`: 512x512x3 canvas normalised to [-1,1]; `vh`, `vw` the valid size.
    pub fn encode(&self, img: &[f32], vh: usize, vw: usize) -> Encoded {
        let valid = |s: usize| (vh.div_ceil(s), vw.div_ceil(s));
        // stage 1: full conv 3->32, stride 2, symmetric pad 1
        let (mut f, mut h, mut w) = conv3(img, CANVAS, CANVAS, 3, &self.w0, 32, 2, 2, Some(1));
        rmsnorm_in_place(&mut f, 32, Some(&self.stem_norms[0]));
        let (a, b) = valid(2);
        mask_grid(&mut f, h, w, 32, a, b);
        let mut c = 32;
        let (mut s4, mut s8, mut s16) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..4 {
            let mut a = f.clone();
            silu_in_place(&mut a);
            let (dw, h2, w2) = dwconv3(&a, h, w, c, &self.stem_dw[j], 2, 2, 1);
            h = h2;
            w = w2;
            let pw = &self.stem_pw[j];
            f = linear(&dw, h * w, &pw.w, pw.out, c, None);
            c = pw.out;
            rmsnorm_in_place(&mut f, c, Some(&self.stem_norms[j + 1]));
            let s = 2 << (j + 1);
            let (a, b) = valid(s);
            mask_grid(&mut f, h, w, c, a, b);
            if matches!(s, 4 | 8 | 16) {
                let mut x = f.clone();
                silu_in_place(&mut x);
                match s {
                    4 => s4 = x,
                    8 => s8 = x,
                    _ => s16 = x,
                }
            }
        }
        let (gh, gw) = valid(32);
        let mut x = f;
        silu_in_place(&mut x); // 16x16x512
        for b in &self.body {
            self.body_block(b, &mut x, gh, gw);
        }
        const T: usize = 260;
        let mut z = vec![0f32; T * D];
        z[..256 * D].copy_from_slice(&x);
        z[256 * D..].copy_from_slice(&self.registers);
        let valids: Vec<bool> = (0..T).map(|p| p >= 256 || ((p >> 4) < gh && (p & 15) < gw)).collect();
        let mut e = vec![0f32; 256 * D];
        for l in 0..12 {
            let zn = rmsnorm(&z[..256 * D], D, Some(&self.level_norms[l]));
            let gl = &self.level_gates[l * D..(l + 1) * D];
            for p in 0..256 {
                for c in 0..D {
                    e[p * D + c] += gl[c] * zn[p * D + c];
                }
            }
            self.attn_block(&self.top[l], &mut z, T, &valids);
        }
        let zf = rmsnorm(&z[..256 * D], D, Some(&self.enc_norm));
        add_in_place(&mut e, &zf);
        mask_grid(&mut e, 16, 16, D, gh, gw);
        Encoded { e, s4, s8, s16, gh, gw, vh, vw }
    }

    fn body_block(&self, b: &Body, x: &mut [f32], gh: usize, gw: usize) {
        let zn = rmsnorm(x, D, Some(&b.mix_norm));
        let (mut v, _, _) = dwconv3(&zn, 16, 16, D, &b.dwv, 1, 1, 1);
        let (g, _, _) = dwconv3(&zn, 16, 16, D, &b.dwg, 1, 1, 1);
        for (i, vv) in v.iter_mut().enumerate() {
            let c = i % D;
            *vv = (*vv + b.bv[c]) * silu(g[i] + b.bg[c]);
        }
        mask_grid(&mut v, 16, 16, D, gh, gw);
        let (mut o, _, _) = dwconv3(&v, 16, 16, D, &b.dwo, 1, 1, 1);
        mask_grid(&mut o, 16, 16, D, gh, gw);
        add_in_place(x, &o);
        let mut hm = b.hconv.run(x, 256, &self.seq);
        mask_grid(&mut hm, 16, 16, D, gh, gw);
        // GRN over valid cells
        let mut r = vec![0f64; D];
        for p in 0..256 {
            for c in 0..D {
                let v = hm[p * D + c] as f64;
                r[c] += v * v;
            }
        }
        let mut mean = 0f64;
        for v in r.iter_mut() {
            *v = (*v + EPS as f64).sqrt();
            mean += *v;
        }
        mean /= D as f64;
        for p in 0..256 {
            if (p >> 4) >= gh || (p & 15) >= gw {
                continue;
            }
            for c in 0..D {
                let h = hm[p * D + c];
                x[p * D + c] += h + b.gamma[c] * h * (r[c] / (mean + EPS as f64)) as f32 + b.beta[c];
            }
        }
    }

    fn attn_block(&self, t: &Top, z: &mut [f32], n: usize, valids: &[bool]) {
        let zn = rmsnorm(z, D, Some(&t.norm));
        let mut q = linear(&zn, n, &t.q.w, 128, D, None);
        let mut k = linear(&zn, n, &t.k.w, 32, D, None);
        let v = linear(&zn, n, &t.v.w, 64, D, None);
        rmsnorm_in_place(&mut q, 32, Some(&t.qn));
        rmsnorm_in_place(&mut k, 32, Some(&t.kn));
        // 2D RoPE: pairs (r, r+16); r < 8 rotate by row, r >= 8 by column
        for p in 0..256 {
            let (row, col) = (p >> 4, p & 15);
            for h in 0..4 {
                rope2d(&mut q[p * 128 + h * 32..p * 128 + h * 32 + 32], row, col);
            }
            rope2d(&mut k[p * 32..p * 32 + 32], row, col);
        }
        let mut att = vec![0f32; n * 256];
        let scale = 1.0 / (32f32).sqrt();
        par::rows(&mut att, 256, n * n * 4 * 96, |p0, blk| {
            let mut sc = vec![0f32; n];
            for (dp, ar) in blk.chunks_exact_mut(256).enumerate() {
                let p = p0 + dp;
                for h in 0..4 {
                    let qh = &q[p * 128 + h * 32..p * 128 + h * 32 + 32];
                    for j in 0..n {
                        sc[j] = if valids[j] { dot(qh, &k[j * 32..j * 32 + 32]) * scale } else { f32::NEG_INFINITY };
                    }
                    softmax_in_place(&mut sc);
                    let ao = &mut ar[h * 64..h * 64 + 64];
                    for j in 0..n {
                        if sc[j] != 0.0 {
                            axpy(ao, sc[j], &v[j * 64..j * 64 + 64]);
                        }
                    }
                }
            }
        });
        let gate = linear(&zn, n, &t.gate.w, 256, D, None);
        for (a, g) in att.iter_mut().zip(&gate) {
            *a *= sigmoid(*g);
        }
        let mut a = linear(&att, n, &t.out.w, D, 256, None);
        rmsnorm_in_place(&mut a, D, Some(&t.post));
        for p in 0..n {
            if valids[p] {
                axpy(&mut z[p * D..(p + 1) * D], t.gate_a, &a[p * D..(p + 1) * D]);
            }
        }
        let hm = t.hconv.run(z, n, &self.seq);
        for p in 0..n {
            if valids[p] {
                add_in_place(&mut z[p * D..(p + 1) * D], &hm[p * D..(p + 1) * D]);
            }
        }
    }

    /// F + PW(DW(SiLU(N(F)))) on a 128-channel map.
    fn res_block(b: &Res, f: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut a = rmsnorm(f, 128, Some(&b.norm));
        silu_in_place(&mut a);
        let (d, _, _) = dwconv3(&a, h, w, 128, &b.dw, 1, 1, 1);
        let mut y = linear(&d, h * w, &b.pw.w, 128, 128, None);
        add_in_place(&mut y, f);
        y
    }

    /// Dense 64x64x128 feature map.
    pub fn neck_forward(&self, enc: &Encoded) -> Vec<f32> {
        let e = linear(&enc.e, 256, &self.coarse.w, 128, D, None);
        let mut t32 = up2(&e, 16, 16, 128);
        add_in_place(&mut t32, &linear(&enc.s16, 32 * 32, &self.mid.w, 128, 256, None));
        let t32 = Self::res_block(&self.neck, &t32, 32, 32);
        let mut f = up2(&t32, 32, 32, 128);
        add_in_place(&mut f, &linear(&enc.s8, 64 * 64, &self.fine.w, 128, 128, None));
        for b in &self.blocks {
            f = Self::res_block(b, &f, 64, 64);
        }
        let mut dm = rmsnorm(&f, 128, Some(&self.fin));
        silu_in_place(&mut dm);
        mask_grid(&mut dm, 64, 64, 128, enc.vh.div_ceil(8), enc.vw.div_ceil(8));
        dm
    }

    /// Dense detection output `[64*64][10]`.
    pub fn detect(&self, dm: &[f32]) -> Vec<f32> {
        linear(dm, 64 * 64, &self.out.w, 10, 128, Some(&self.bias))
    }

    fn mask_head(h: &MaskHead, f: &[f32], hh: usize, ww: usize) -> Vec<f32> {
        let mut f = f.to_vec();
        for b in &h.blocks {
            f = Self::res_block(b, &f, hh, ww);
        }
        let mut a = rmsnorm(&f, 128, Some(&h.fin));
        silu_in_place(&mut a);
        linear(&a, hh * ww, &h.proj.w, 32, 128, None)
    }

    /// Mask coefficients, `[64*64][32]`.
    pub fn coefficients(&self, dm: &[f32]) -> Vec<f32> {
        Self::mask_head(&self.coef, dm, 64, 64)
    }

    /// Mask prototypes, `[128*128][32]`.
    pub fn prototypes(&self, dm: &[f32], enc: &Encoded) -> Vec<f32> {
        let mut f = up2(dm, 64, 64, 128);
        add_in_place(&mut f, &linear(&enc.s4, 128 * 128, &self.edge.w, 128, 64, None));
        Self::mask_head(&self.proto, &f, 128, 128)
    }

    /// Box-reference slot for normalised canvas box b = [x1,y1,x2,y2] with centre cell (ci,cj) on the 64 grid.
    pub fn slot_vector(&self, dm: &[f32], enc: &Encoded, b: [f32; 4], ci: usize, cj: usize, ref_embedding: &[f32]) -> Vec<f32> {
        let mut f = vec![0f32; 192];
        f[..128].copy_from_slice(&dm[(ci * 64 + cj) * 128..(ci * 64 + cj + 1) * 128]);
        // per corner 8 sin then 8 cos of 2^r * pi * b
        for c in 0..4 {
            let o = 128 + c * 16;
            for r in 0..8 {
                let a = (1u32 << r) as f64 * std::f64::consts::PI * b[c] as f64;
                f[o + r] = a.sin() as f32;
                f[o + 8 + r] = a.cos() as f32;
            }
        }
        let mut z = linear(&f, 1, &self.slot_proj.w, D, 192, None);
        // average the valid coarse cells whose centres lie in the box, plus the centre cell
        let mut cells = Vec::new();
        for i in 0..enc.gh {
            for j in 0..enc.gw {
                let y = (i as f32 + 0.5) / 16.0;
                let x = (j as f32 + 0.5) / 16.0;
                if x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3] {
                    cells.push(i * 16 + j);
                }
            }
        }
        let centre = (enc.gh - 1).min(ci >> 2) * 16 + (enc.gw - 1).min(cj >> 2);
        if !cells.contains(&centre) {
            cells.push(centre);
        }
        let mut pooled = vec![0f32; D];
        for &p in &cells {
            add_in_place(&mut pooled, &enc.e[p * D..(p + 1) * D]);
        }
        let n = cells.len() as f32;
        for v in pooled.iter_mut() {
            *v /= n;
        }
        add_in_place(&mut z, &linear(&pooled, 1, &self.slot_pool.w, D, D, None));
        let mut s = rmsnorm(&z, D, Some(&self.slot_norm));
        let sq = (D as f32).sqrt();
        axpy(&mut s, sq, ref_embedding);
        s
    }
}
