//! Mirage language decoder: 8 Simple-Attention layers on 4 mHC residual lanes,
//! engram fusion at layers 3 and 7, causal self-attention with depthwise taps,
//! visual cross-attention into the image memory, learned Hadamard MLP.
//!
//! The weights are immutable; everything that changes while decoding lives in
//! [`DecState`], so one model can decode several sequences at once.

use crate::cact::{Cact, Mat};
use crate::ops::*;
use crate::{err, Result};

const D: usize = 512;
const L: usize = 8;
const LANES: usize = 4;
const HQ: usize = 8;
const HKV: usize = 2;
const QK: usize = 48;
const VD: usize = 64;
const TAPS: usize = 3;
const SLOTS: u32 = 18432;
const ORDERS: [usize; 2] = [2, 3];
const EHEADS: usize = 2;
const SUB: usize = 128;
const ETAPS: usize = 4;
const EDIL: usize = 3;
const SITES: [usize; 2] = [3, 7];
const SEED: u32 = 0x9e37_79b9;
const PRIME: u32 = 0x0100_0193;

struct Layer {
    n0: Vec<f32>,
    q: Mat,
    k: Mat,
    v: Mat,
    qt: Vec<f32>,
    kt: Vec<f32>,
    vt: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    gate: Mat,
    out: Mat,
    post: Vec<f32>,
    ag: f32,
    cn: Vec<f32>,
    cq: Mat,
    ck: Mat,
    cv: Mat,
    cgate: Mat,
    cout: Mat,
    cqn: Vec<f32>,
    ckn: Vec<f32>,
    cpost: Vec<f32>,
    cg: f32,
    pre: Vec<f32>,
    d1: Vec<f32>,
    d2: Vec<f32>,
    b2: Vec<f32>,
    d3: Vec<f32>,
    d4: Vec<f32>,
    w: [(Vec<f32>, Vec<f32>); 3],
    cond_v: Vec<f32>,
    cond_u: Vec<f32>,
}

struct Engram {
    table: String,
    key: Mat,
    val: Mat,
    taps: Vec<f32>,
}

/// Raw `[out][in]` matrix used row-block-wise (mHC phi projections).
struct Rows {
    w: Vec<f32>,
    inp: usize,
}

impl Rows {
    fn rows(&self, x: &[f32], r0: usize, r1: usize) -> Vec<f32> {
        linear(x, 1, &self.w[r0 * self.inp..r1 * self.inp], r1 - r0, self.inp, None)
    }
}

pub struct Decoder {
    embed: Vec<f32>,
    /// vocabulary size (text pieces plus reference tokens)
    pub v: usize,
    layers: Vec<Layer>,
    p1: Vec<usize>,
    p2: Vec<usize>,
    a_pre: Vec<f32>,
    a_post: Vec<f32>,
    a_res: Vec<f32>,
    b_pre: Vec<f32>,
    b_post: Vec<f32>,
    b_res: Vec<f32>,
    engrams: Vec<Engram>,
    final_norm: Vec<f32>,
    phi_pre: Rows,
    phi_post: Rows,
    phi_res: Rows,
}

#[derive(Default)]
struct LayerCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    rq: Vec<Vec<f32>>,
    rk: Vec<Vec<f32>>,
    rv: Vec<Vec<f32>>,
}

/// Per-sequence decoder state: image memory and caches.
pub struct DecState {
    t: usize,
    tokens: Vec<u32>,
    cache: Vec<LayerCache>,
    ev: [Vec<Vec<f32>>; 2],
    mn: usize,
    mvalid: Vec<bool>,
    cross: Vec<(Vec<f32>, Vec<f32>)>,
}

fn rms_unit(x: &[f32]) -> Vec<f32> {
    rmsnorm(x, x.len(), None)
}

fn rope(x: &mut [f32], pos: usize, q: usize, theta: f64) {
    let h = q >> 1;
    for r in 0..h {
        let w = pos as f64 * theta.powf(-2.0 * r as f64 / q as f64);
        let (c, s) = (w.cos() as f32, w.sin() as f32);
        let (a, b) = (x[r], x[r + h]);
        x[r] = a * c - b * s;
        x[r + h] = b * c + a * s;
    }
}

/// out[k][l] = sum_ij z[i][j] a[i][k] b[j][l], z 16x32
fn kron(z: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    let mut t = [0f32; 512];
    for i in 0..16 {
        for l in 0..32 {
            let mut s = 0f32;
            for j in 0..32 {
                s += z[i * 32 + j] * b[j * 32 + l];
            }
            t[i * 32 + l] = s;
        }
    }
    for k in 0..16 {
        for l in 0..32 {
            let mut s = 0f32;
            for i in 0..16 {
                s += a[i * 16 + k] * t[i * 32 + l];
            }
            out[k * 32 + l] = s;
        }
    }
}

fn mv(m: &Mat, x: &[f32]) -> Vec<f32> {
    linear(x, 1, &m.w, m.out, m.inp, None)
}

impl Decoder {
    pub fn new(c: &Cact) -> Result<Self> {
        let embed = c.get("embedding/embedding")?;
        let v = c.rec("embedding/embedding", None)?.shape[0];
        let g = |n: &str, l: usize| c.get_l(&format!("dec/{n}"), Some(l));
        let dn = |n: &str, l: usize| c.dense_l(&format!("dec/{n}"), Some(l));
        let mut layers = Vec::with_capacity(L);
        for l in 0..L {
            let w = |s: usize| -> Result<(Vec<f32>, Vec<f32>)> { Ok((g(&format!("hadamard_mlp/w{s}a"), l)?, g(&format!("hadamard_mlp/w{s}b"), l)?)) };
            layers.push(Layer {
                n0: g("ZCRMSNorm_0/scale", l)?,
                q: dn("self_attn/q_proj/kernel", l)?,
                k: dn("self_attn/k_proj/kernel", l)?,
                v: dn("self_attn/v_proj/kernel", l)?,
                qt: g("self_attn/q_taps", l)?,
                kt: g("self_attn/k_taps", l)?,
                vt: g("self_attn/v_taps", l)?,
                qn: g("self_attn/q_norm/scale", l)?,
                kn: g("self_attn/k_norm/scale", l)?,
                gate: dn("self_attn/gate_proj/kernel", l)?,
                out: dn("self_attn/out_proj/kernel", l)?,
                post: g("post_attn_norm/scale", l)?,
                ag: sigmoid(g("attn_gate", l)?[0]),
                cn: g("cross_norm/scale", l)?,
                cq: dn("cross_attn/q_proj/kernel", l)?,
                ck: dn("cross_attn/k_proj/kernel", l)?,
                cv: dn("cross_attn/v_proj/kernel", l)?,
                cgate: dn("cross_attn/gate_proj/kernel", l)?,
                cout: dn("cross_attn/out_proj/kernel", l)?,
                cqn: g("cross_attn/q_norm/scale", l)?,
                ckn: g("cross_attn/k_norm/scale", l)?,
                cpost: g("post_cross_norm/scale", l)?,
                cg: sigmoid(g("cross_gate", l)?[0]),
                pre: g("pre_hada_norm/scale", l)?,
                d1: g("hadamard_mlp/d1", l)?,
                d2: g("hadamard_mlp/d2", l)?,
                b2: g("hadamard_mlp/b2", l)?,
                d3: g("hadamard_mlp/d3", l)?,
                d4: g("hadamard_mlp/d4", l)?,
                w: [w(1)?, w(2)?, w(3)?],
                cond_v: g("hadamard_mlp/cond_v", l)?,
                cond_u: g("hadamard_mlp/cond_u", l)?,
            });
        }
        let perm = |n: &str| -> Result<Vec<usize>> {
            let p: Vec<usize> = c.get(n)?.iter().map(|&v| v as usize).collect();
            if p.len() != D || p.iter().any(|&i| i >= D) {
                return err(format!("{n}: bad permutation"));
            }
            Ok(p)
        };
        let rows = |n: &str| -> Result<Rows> { Ok(Rows { w: c.get(n)?, inp: c.rec(n, None)?.shape[1] }) };
        let engrams = (0..2)
            .map(|s| -> Result<Engram> {
                let table = format!("engrams_{s}/embedding");
                c.rows(&table, None)?;
                Ok(Engram {
                    table,
                    key: c.dense(&format!("engrams_{s}/key_proj/kernel"))?,
                    val: c.dense(&format!("engrams_{s}/value_proj/kernel"))?,
                    taps: c.get(&format!("engrams_{s}/taps"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Decoder {
            embed,
            v,
            layers,
            p1: perm("hada_perm1")?,
            p2: perm("hada_perm2")?,
            a_pre: c.get("stack/mhc_a_pre")?,
            a_post: c.get("stack/mhc_a_post")?,
            a_res: c.get("stack/mhc_a_res")?,
            b_pre: c.get("stack/mhc_b_pre")?,
            b_post: c.get("stack/mhc_b_post")?,
            b_res: c.get("stack/mhc_b_res")?,
            engrams,
            final_norm: c.get("stack/final_norm/scale")?,
            phi_pre: rows("stack/mhc_phi_pre")?,
            phi_post: rows("stack/mhc_phi_post")?,
            phi_res: rows("stack/mhc_phi_res")?,
        })
    }

    pub fn embedding(&self, id: usize) -> &[f32] {
        &self.embed[id * D..(id + 1) * D]
    }

    /// Fresh state over image memory `m` (`mn` rows of 512) with its validity mask.
    pub fn start(&self, m: &[f32], mn: usize, valid: Vec<bool>) -> DecState {
        let cross = self
            .layers
            .iter()
            .map(|ly| {
                let mut k = linear(m, mn, &ly.ck.w, ly.ck.out, D, None);
                rmsnorm_in_place(&mut k, QK, Some(&ly.ckn));
                let v = linear(m, mn, &ly.cv.w, ly.cv.out, D, None);
                (k, v)
            })
            .collect();
        DecState {
            t: 0,
            tokens: Vec::new(),
            cache: (0..L).map(|_| LayerCache::default()).collect(),
            ev: [Vec::new(), Vec::new()],
            mn,
            mvalid: valid,
            cross,
        }
    }

    fn engram_kv(&self, c: &Cact, st: &mut DecState, s: usize) -> (Vec<f32>, Vec<f32>) {
        let e = &self.engrams[s];
        let rows = c.rows(&e.table, None).expect("engram table checked at load");
        let (t, toks) = (st.t, &st.tokens);
        let mut vec = vec![0f32; 4 * SUB];
        let mut tmp = vec![0f32; rows.ip];
        let mut table = 0usize;
        for (oi, &order) in ORDERS.iter().enumerate() {
            for h in 0..EHEADS {
                if t + 1 >= order {
                    let mut acc = SEED.wrapping_mul((oi * EHEADS + h + 1) as u32);
                    for j in 0..order {
                        let tok = if t >= j { toks[t - j] } else { 0 };
                        acc = (acc ^ tok).wrapping_mul(PRIME);
                    }
                    acc ^= acc >> 15;
                    let row = table * SLOTS as usize + (acc % SLOTS) as usize;
                    rows.row_into(row, &mut vec[table * SUB..(table + 1) * SUB], &mut tmp);
                }
                table += 1;
            }
        }
        let k = mv(&e.key, &vec);
        let v = mv(&e.val, &vec);
        st.ev[s].push(v);
        let mut vt = vec![0f32; D];
        for j in 0..ETAPS {
            if t < j * EDIL {
                break;
            }
            let vv = &st.ev[s][t - j * EDIL];
            for c in 0..D {
                vt[c] += e.taps[j * D + c] * vv[c];
            }
        }
        (k, vt)
    }

    fn self_attn(&self, ly: &Layer, cache: &mut LayerCache, t: usize, x: &[f32]) -> Vec<f32> {
        cache.rq.push(mv(&ly.q, x));
        cache.rk.push(mv(&ly.k, x));
        cache.rv.push(mv(&ly.v, x));
        let tap = |hist: &[Vec<f32>], taps: &[f32], n: usize| {
            let mut out = vec![0f32; n];
            for j in 0..TAPS.min(t + 1) {
                let z = &hist[t - j];
                for i in 0..n {
                    out[i] += taps[j * n + i] * z[i];
                }
            }
            out
        };
        let mut q = tap(&cache.rq, &ly.qt, HQ * QK);
        let mut k = tap(&cache.rk, &ly.kt, HKV * QK);
        let v = tap(&cache.rv, &ly.vt, HKV * VD);
        if t >= TAPS {
            cache.rq[t - TAPS] = Vec::new();
            cache.rk[t - TAPS] = Vec::new();
            cache.rv[t - TAPS] = Vec::new();
        }
        for h in q.chunks_exact_mut(QK) {
            rmsnorm_in_place(h, QK, Some(&ly.qn));
            rope(h, t, QK, 1e5);
        }
        for h in k.chunks_exact_mut(QK) {
            rmsnorm_in_place(h, QK, Some(&ly.kn));
            rope(h, t, QK, 1e5);
        }
        cache.k.push(k);
        cache.v.push(v);
        let mut att = vec![0f32; HQ * VD];
        let mut sc = vec![0f32; t + 1];
        let scale = 1.0 / (QK as f32).sqrt();
        for h in 0..HQ {
            let g = h / (HQ / HKV);
            let qh = &q[h * QK..(h + 1) * QK];
            for j in 0..=t {
                sc[j] = dot(qh, &cache.k[j][g * QK..(g + 1) * QK]) * scale;
            }
            softmax_in_place(&mut sc);
            for j in 0..=t {
                axpy(&mut att[h * VD..(h + 1) * VD], sc[j], &cache.v[j][g * VD..(g + 1) * VD]);
            }
        }
        let gt = mv(&ly.gate, x);
        for (a, g) in att.iter_mut().zip(&gt) {
            *a *= sigmoid(*g);
        }
        mv(&ly.out, &att)
    }

    fn cross_attn(&self, ly: &Layer, st: &DecState, l: usize, x: &[f32]) -> Vec<f32> {
        let (ck, cv) = &st.cross[l];
        let mut q = mv(&ly.cq, x);
        rmsnorm_in_place(&mut q, QK, Some(&ly.cqn));
        let mut att = vec![0f32; HQ * VD];
        let mut sc = vec![0f32; st.mn];
        let scale = 1.0 / (QK as f32).sqrt();
        for h in 0..HQ {
            let qh = &q[h * QK..(h + 1) * QK];
            for j in 0..st.mn {
                sc[j] = if st.mvalid[j] { dot(qh, &ck[(j * HQ + h) * QK..(j * HQ + h + 1) * QK]) * scale } else { f32::NEG_INFINITY };
            }
            softmax_in_place(&mut sc);
            for j in 0..st.mn {
                if sc[j] != 0.0 {
                    axpy(&mut att[h * VD..(h + 1) * VD], sc[j], &cv[(j * HQ + h) * VD..(j * HQ + h + 1) * VD]);
                }
            }
        }
        let gt = mv(&ly.cgate, x);
        for (a, g) in att.iter_mut().zip(&gt) {
            *a *= sigmoid(*g);
        }
        mv(&ly.cout, &att)
    }

    fn hmlp(&self, ly: &Layer, x: &[f32]) -> Vec<f32> {
        // cond = 1 + softmax(x @ condV) @ condU
        let mut r = [0f32; 8];
        for (k, rk) in r.iter_mut().enumerate() {
            let mut s = 0f32;
            for i in 0..D {
                s += x[i] * ly.cond_v[i * 8 + k];
            }
            *rk = s;
        }
        softmax_in_place(&mut r);
        let mut cond = vec![1f32; D];
        for (k, &w) in r.iter().enumerate() {
            axpy(&mut cond, w, &ly.cond_u[k * D..(k + 1) * D]);
        }
        let (mut z, mut y, mut t) = (vec![0f32; D], vec![0f32; D], vec![0f32; D]);
        for i in 0..D {
            z[i] = ly.d1[i] * x[i];
        }
        kron(&z, &ly.w[0].0, &ly.w[0].1, &mut y);
        for i in 0..D {
            t[i] = y[self.p1[i]];
        }
        for i in 0..D {
            z[i] = silu(ly.d2[i] * cond[i] * t[i] + ly.b2[i]);
        }
        kron(&z, &ly.w[1].0, &ly.w[1].1, &mut y);
        for i in 0..D {
            t[i] = y[self.p2[i]];
        }
        for i in 0..D {
            z[i] = ly.d3[i] * t[i];
        }
        kron(&z, &ly.w[2].0, &ly.w[2].1, &mut y);
        for i in 0..D {
            y[i] *= ly.d4[i];
        }
        y
    }

    fn block(&self, c: &Cact, st: &mut DecState, l: usize, u: &[f32]) -> Vec<f32> {
        let ly = &self.layers[l];
        let mut x = u.to_vec();
        if let Some(site) = SITES.iter().position(|&s| s == l) {
            let (k, v) = self.engram_kv(c, st, site);
            let a = rms_unit(&x);
            let b = rms_unit(&k);
            let alpha = sigmoid(dot(&a, &b) / (D as f32).sqrt());
            axpy(&mut x, alpha, &v);
        }
        let t = st.t;
        let mut h = self.self_attn(ly, &mut st.cache[l], t, &rmsnorm(&x, D, Some(&ly.n0)));
        rmsnorm_in_place(&mut h, D, Some(&ly.post));
        axpy(&mut x, ly.ag, &h);
        let mut h = self.cross_attn(ly, st, l, &rmsnorm(&x, D, Some(&ly.cn)));
        rmsnorm_in_place(&mut h, D, Some(&ly.cpost));
        axpy(&mut x, ly.cg, &h);
        let h = self.hmlp(ly, &rmsnorm(&x, D, Some(&ly.pre)));
        add_in_place(&mut x, &h);
        x
    }

    /// Feed one token; returns the final (normalised) hidden state.
    pub fn step(&self, c: &Cact, st: &mut DecState, token: u32) -> Vec<f32> {
        st.tokens.push(token);
        let e = self.embedding(token as usize);
        let sq = (D as f32).sqrt();
        let mut x = vec![0f32; LANES * D];
        for n in 0..LANES {
            for i in 0..D {
                x[n * D + i] = sq * e[i];
            }
        }
        for l in 0..L {
            let xh = rms_unit(&x);
            let lane = l % LANES;
            let mut u = vec![0f32; D];
            let mut hpost = [0f32; LANES];
            let ppre = self.phi_pre.rows(&xh, l * LANES, (l + 1) * LANES);
            let ppost = self.phi_post.rows(&xh, l * LANES, (l + 1) * LANES);
            let pres = self.phi_res.rows(&xh, l * 16, (l + 1) * 16);
            for n in 0..LANES {
                let pre = sigmoid(self.a_pre[l] * ppre[n] + self.b_pre[l * LANES + n] + if n == lane { 4.0 } else { -4.0 });
                axpy(&mut u, pre, &x[n * D..(n + 1) * D]);
                hpost[n] = 2.0 * sigmoid(self.a_post[l] * ppost[n] + self.b_post[l * LANES + n] + if n == lane { 0.0 } else { -4.0 });
            }
            // Sinkhorn over the 4x4 residual mixing logits
            let mut lg = [0f64; 16];
            for ij in 0..16 {
                lg[ij] = (self.a_res[l] * pres[ij] + self.b_res[l * 16 + ij]) as f64;
            }
            for _ in 0..20 {
                for i in 0..4 {
                    let m = (0..4).map(|j| lg[i * 4 + j]).fold(f64::NEG_INFINITY, f64::max);
                    let ls = m + (0..4).map(|j| (lg[i * 4 + j] - m).exp()).sum::<f64>().ln();
                    for j in 0..4 {
                        lg[i * 4 + j] -= ls;
                    }
                }
                for j in 0..4 {
                    let m = (0..4).map(|i| lg[i * 4 + j]).fold(f64::NEG_INFINITY, f64::max);
                    let ls = m + (0..4).map(|i| (lg[i * 4 + j] - m).exp()).sum::<f64>().ln();
                    for i in 0..4 {
                        lg[i * 4 + j] -= ls;
                    }
                }
            }
            let mut y = self.block(c, st, l, &u);
            for i in 0..D {
                y[i] -= u[i];
            }
            let mut xn = vec![0f32; LANES * D];
            for i in 0..4 {
                let row = &mut xn[i * D..(i + 1) * D];
                for j in 0..4 {
                    axpy(row, lg[i * 4 + j].exp() as f32, &x[j * D..(j + 1) * D]);
                }
                axpy(row, hpost[i], &y);
            }
            x = xn;
        }
        let mut h = vec![0f32; D];
        for n in 0..LANES {
            for i in 0..D {
                h[i] += x[n * D + i] / LANES as f32;
            }
        }
        st.t += 1;
        rmsnorm_in_place(&mut h, D, Some(&self.final_norm));
        h
    }

    /// Next-token logits (tied embedding).
    pub fn logits(&self, h: &[f32], out: &mut [f32]) {
        linear_into(h, 1, &self.embed, self.v, D, None, out);
    }
}
