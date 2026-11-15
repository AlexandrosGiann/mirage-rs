//! Numeric kernels. Feature maps are HWC (row-major), token sequences are
//! `[T][C]`, dense weights are row-major `[out][in]`.

use crate::par;

pub const EPS: f32 = 1e-6;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

#[inline]
pub fn silu(x: f32) -> f32 {
    (x as f64 / (1.0 + (-(x as f64)).exp())) as f32
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    let x = x as f64;
    let gc = (2.0 / std::f64::consts::PI).sqrt();
    (0.5 * x * (1.0 + (gc * (x + 0.044715 * x * x * x)).tanh())) as f32
}

pub fn silu_in_place(a: &mut [f32]) {
    for v in a {
        *v = silu(*v);
    }
}

/// Dot product with eight independent accumulators (vectorises well).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

/// Zero-centred-scale RMSNorm over rows of `c`: (1+s)*x/sqrt(mean(x^2)+eps).
pub fn rmsnorm_in_place(x: &mut [f32], c: usize, scale: Option<&[f32]>) {
    for row in x.chunks_exact_mut(c) {
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = (1.0 / (ss / c as f64 + EPS as f64).sqrt()) as f32;
        match scale {
            Some(s) => {
                for (v, &k) in row.iter_mut().zip(s) {
                    *v = (1.0 + k) * *v * inv;
                }
            }
            None => {
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }
    }
}

pub fn rmsnorm(x: &[f32], c: usize, scale: Option<&[f32]>) -> Vec<f32> {
    let mut y = x.to_vec();
    rmsnorm_in_place(&mut y, c, scale);
    y
}

/// `Y[n][o] = sum_i X[n][i] W[o][i] (+ b[o])`.
pub fn linear(x: &[f32], n: usize, w: &[f32], out: usize, inp: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0f32; n * out];
    linear_into(x, n, w, out, inp, bias, &mut y);
    y
}

pub fn linear_into(x: &[f32], n: usize, w: &[f32], out: usize, inp: usize, bias: Option<&[f32]>, y: &mut [f32]) {
    let work = n * out * inp;
    let b = |o: usize| bias.map_or(0.0, |b| b[o]);
    if n == 1 {
        // one vector: split the output features
        par::rows(&mut y[..out], 1, work, |o0, blk| {
            for (k, v) in blk.iter_mut().enumerate() {
                let o = o0 + k;
                *v = dot(&x[..inp], &w[o * inp..(o + 1) * inp]) + b(o);
            }
        });
        return;
    }
    par::rows(&mut y[..n * out], out, work, |r0, blk| {
        for (k, yr) in blk.chunks_exact_mut(out).enumerate() {
            let xr = &x[(r0 + k) * inp..(r0 + k + 1) * inp];
            for (o, v) in yr.iter_mut().enumerate() {
                *v = dot(xr, &w[o * inp..(o + 1) * inp]) + b(o);
            }
        }
    });
}

/// Depthwise 3x3 conv (HWC), kernel `[9][C]`, strides (sh, sw), `pad` zeros before the first row/col.
pub fn dwconv3(x: &[f32], h: usize, w: usize, c: usize, k: &[f32], sh: usize, sw: usize, pad: usize) -> (Vec<f32>, usize, usize) {
    dwconv(x, h, w, c, k, 3, 3, sh, sw, pad, pad)
}

/// Depthwise kh x kw conv with explicit leading padding.
#[allow(clippy::too_many_arguments)]
pub fn dwconv(x: &[f32], h: usize, w: usize, c: usize, k: &[f32], kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> (Vec<f32>, usize, usize) {
    let ho = h.div_ceil(sh);
    let wo = w.div_ceil(sw);
    let mut y = vec![0f32; ho * wo * c];
    par::rows(&mut y, wo * c, ho * wo * c * kh * kw, |i0, blk| {
        for (di, yr) in blk.chunks_exact_mut(wo * c).enumerate() {
            let i = i0 + di;
            for dy in 0..kh {
                let yy = (i * sh + dy) as isize - ph as isize;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                for j in 0..wo {
                    let yo = &mut yr[j * c..(j + 1) * c];
                    for dx in 0..kw {
                        let xx = (j * sw + dx) as isize - pw as isize;
                        if xx < 0 || xx >= w as isize {
                            continue;
                        }
                        let xo = (yy as usize * w + xx as usize) * c;
                        let ko = (dy * kw + dx) * c;
                        let (xs, ks) = (&x[xo..xo + c], &k[ko..ko + c]);
                        for ((o, a), b) in yo.iter_mut().zip(xs).zip(ks) {
                            *o += a * b;
                        }
                    }
                }
            }
        }
    });
    (y, ho, wo)
}

/// SAME padding (leading half) for a kernel of size `k`.
pub fn same_pad(n: usize, s: usize, k: usize) -> usize {
    let o = n.div_ceil(s);
    (((o - 1) * s + k).saturating_sub(n)) >> 1
}

/// Full 3x3 conv, kernel HWIO flattened `[(kh*3+kw)*Ci + ci][Co]`, strides (sh, sw),
/// leading padding `pad` (SAME when None).
#[allow(clippy::too_many_arguments)]
pub fn conv3(x: &[f32], h: usize, w: usize, ci: usize, k: &[f32], co: usize, sh: usize, sw: usize, pad: Option<usize>) -> (Vec<f32>, usize, usize) {
    let ho = h.div_ceil(sh);
    let wo = w.div_ceil(sw);
    let ph = pad.unwrap_or_else(|| same_pad(h, sh, 3));
    let pw = pad.unwrap_or_else(|| same_pad(w, sw, 3));
    let mut y = vec![0f32; ho * wo * co];
    par::rows(&mut y, wo * co, ho * wo * co * ci * 9, |i0, blk| {
        for (di, yr) in blk.chunks_exact_mut(wo * co).enumerate() {
            let i = i0 + di;
            for j in 0..wo {
                let yo = &mut yr[j * co..(j + 1) * co];
                for dy in 0..3 {
                    let yy = (i * sh + dy) as isize - ph as isize;
                    if yy < 0 || yy >= h as isize {
                        continue;
                    }
                    for dx in 0..3 {
                        let xx = (j * sw + dx) as isize - pw as isize;
                        if xx < 0 || xx >= w as isize {
                            continue;
                        }
                        let xo = (yy as usize * w + xx as usize) * ci;
                        for c in 0..ci {
                            let v = x[xo + c];
                            if v == 0.0 {
                                continue;
                            }
                            let ko = ((dy * 3 + dx) * ci + c) * co;
                            axpy(yo, v, &k[ko..ko + co]);
                        }
                    }
                }
            }
        }
    });
    (y, ho, wo)
}

/// Nearest-neighbour x2 upsample (HWC).
pub fn up2(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let w2 = 2 * w;
    let mut y = vec![0f32; 4 * h * w * c];
    for i in 0..2 * h {
        for j in 0..w2 {
            let s = ((i >> 1) * w + (j >> 1)) * c;
            let d = (i * w2 + j) * c;
            y[d..d + c].copy_from_slice(&x[s..s + c]);
        }
    }
    y
}

pub fn add_in_place(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Zero rows outside the valid vh x vw region of an h x w grid.
pub fn mask_grid(x: &mut [f32], h: usize, w: usize, c: usize, vh: usize, vw: usize) {
    if vh >= h && vw >= w {
        return;
    }
    for i in 0..h {
        for j in 0..w {
            if i < vh && j < vw {
                continue;
            }
            x[(i * w + j) * c..(i * w + j + 1) * c].fill(0.0);
        }
    }
}

/// Softmax in place; `-inf` entries become 0.
pub fn softmax_in_place(a: &mut [f32]) {
    let m = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        a.fill(0.0);
        return;
    }
    let mut s = 0f64;
    for v in a.iter_mut() {
        let e = ((*v - m) as f64).exp();
        *v = e as f32;
        s += e;
    }
    let inv = (1.0 / s) as f32;
    for v in a.iter_mut() {
        *v *= inv;
    }
}
