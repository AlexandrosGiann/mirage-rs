//! Mask post-processing: rasterisation in original-image pixels, boundary
//! tracing to polygons, Douglas-Peucker simplification and COCO RLE.

use std::collections::HashMap;

/// `logits` over the 128x128 canvas grid (4 px cells); `rx`, `ry` canvas px per
/// raster px. Returns a W x H 0/1 mask restricted to `bx` (raster pixels).
pub fn rasterise(logits: &[f32], rx: f32, ry: f32, w: usize, h: usize, bx: [f32; 4], threshold: f32) -> Vec<u8> {
    let thr = (threshold / (1.0 - threshold)).ln();
    let mut out = vec![0u8; w * h];
    let x0 = bx[0].floor().max(0.0) as usize;
    let x1 = (bx[2].ceil().max(0.0) as usize).min(w);
    let y0 = bx[1].floor().max(0.0) as usize;
    let y1 = (bx[3].ceil().max(0.0) as usize).min(h);
    for y in y0..y1 {
        let gy = (((y as f64 + 0.5) * ry as f64) / 4.0 - 0.5).clamp(0.0, 127.0);
        let iy = gy.floor() as usize;
        let ty = gy - iy as f64;
        let iy1 = (iy + 1).min(127);
        for x in x0..x1 {
            let gx = (((x as f64 + 0.5) * rx as f64) / 4.0 - 0.5).clamp(0.0, 127.0);
            let ix = gx.floor() as usize;
            let tx = gx - ix as f64;
            let ix1 = (ix + 1).min(127);
            let l = |i: usize, j: usize| logits[i * 128 + j] as f64;
            let v = (l(iy, ix) * (1.0 - tx) + l(iy, ix1) * tx) * (1.0 - ty) + (l(iy1, ix) * (1.0 - tx) + l(iy1, ix1) * tx) * ty;
            if v > thr as f64 {
                out[y * w + x] = 1;
            }
        }
    }
    out
}

/// COCO uncompressed RLE (column-major): (size [H, W], counts).
pub fn rle(mask: &[u8], w: usize, h: usize) -> ([usize; 2], Vec<usize>) {
    let mut counts = Vec::new();
    let (mut cur, mut run) = (0u8, 0usize);
    for x in 0..w {
        for y in 0..h {
            let v = mask[y * w + x];
            if v == cur {
                run += 1;
            } else {
                counts.push(run);
                run = 1;
                cur = v;
            }
        }
    }
    counts.push(run);
    ([h, w], counts)
}

/// Outer boundaries of 4-connected regions as closed loops on the pixel-corner
/// grid, with their (positive) areas, largest first.
fn trace_outer(mask: &[u8], w: usize, h: usize) -> Vec<(Vec<(i64, i64)>, f64)> {
    let vw = w + 1;
    let mut next: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut order: Vec<usize> = Vec::new();
    let mut add = |a: usize, b: usize, next: &mut HashMap<usize, Vec<usize>>| {
        next.entry(a)
            .or_insert_with(|| {
                order.push(a);
                Vec::new()
            })
            .push(b);
    };
    let at = |x: isize, y: isize| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && mask[y as usize * w + x as usize] != 0;
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] == 0 {
                continue;
            }
            let (xi, yi) = (x as isize, y as isize);
            if !at(xi, yi - 1) {
                add(y * vw + x, y * vw + x + 1, &mut next);
            }
            if !at(xi + 1, yi) {
                add(y * vw + x + 1, (y + 1) * vw + x + 1, &mut next);
            }
            if !at(xi, yi + 1) {
                add((y + 1) * vw + x + 1, (y + 1) * vw + x, &mut next);
            }
            if !at(xi - 1, yi) {
                add((y + 1) * vw + x, y * vw + x, &mut next);
            }
        }
    }
    let mut loops = Vec::new();
    let guard_max = 4 * (w + 1) * (h + 1);
    for &start in &order {
        while next.get(&start).is_some_and(|l| !l.is_empty()) {
            let mut lp = Vec::new();
            let mut v = start;
            let mut guard = 0;
            loop {
                lp.push(v);
                let l = next.get_mut(&v).unwrap();
                let n = l.pop().unwrap();
                if l.is_empty() {
                    next.remove(&v);
                }
                v = n;
                guard += 1;
                if guard > guard_max || v == start || !next.contains_key(&v) {
                    break;
                }
            }
            if lp.len() >= 4 {
                loops.push(lp.iter().map(|&p| ((p % vw) as i64, (p / vw) as i64)).collect::<Vec<_>>());
            }
        }
    }
    let area = |pts: &[(i64, i64)]| -> f64 {
        let mut s = 0i64;
        for i in 0..pts.len() {
            let (x1, y1) = pts[i];
            let (x2, y2) = pts[(i + 1) % pts.len()];
            s += x1 * y2 - x2 * y1;
        }
        s as f64 / 2.0
    };
    let mut out: Vec<_> = loops.into_iter().map(|p| {
        let a = area(&p);
        (p, a)
    }).filter(|(_, a)| *a > 0.0).collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out
}

fn dp(pts: &[(i64, i64)], tol: f64) -> Vec<(i64, i64)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        let (ax, ay) = (pts[a].0 as f64, pts[a].1 as f64);
        let (dx, dy) = (pts[b].0 as f64 - ax, pts[b].1 as f64 - ay);
        let len = { let l = dx.hypot(dy); if l == 0.0 { 1.0 } else { l } };
        let (mut md, mut mi) = (-1f64, 0usize);
        for (i, p) in pts.iter().enumerate().take(b).skip(a + 1) {
            let d = (dy * (p.0 as f64 - ax) - dx * (p.1 as f64 - ay)).abs() / len;
            if d > md {
                md = d;
                mi = i;
            }
        }
        if md > tol && mi > 0 {
            keep[mi] = true;
            stack.push((a, mi));
            stack.push((mi, b));
        }
    }
    pts.iter().zip(&keep).filter(|(_, &k)| k).map(|(p, _)| *p).collect()
}

fn simplify_closed(pts: &[(i64, i64)], tol: f64) -> Vec<(i64, i64)> {
    if pts.len() <= 4 {
        return pts.to_vec();
    }
    let (mut far, mut fd) = (0, -1i64);
    for (i, p) in pts.iter().enumerate().skip(1) {
        let d = (p.0 - pts[0].0).pow(2) + (p.1 - pts[0].1).pow(2);
        if d > fd {
            fd = d;
            far = i;
        }
    }
    let a = dp(&pts[..=far], tol);
    let mut tail = pts[far..].to_vec();
    tail.push(pts[0]);
    let b = dp(&tail, tol);
    let mut out = a[..a.len() - 1].to_vec();
    out.extend_from_slice(&b[..b.len() - 1]);
    out
}

/// Polygons in 0..1000 coordinates as flat [x, y, ...] lists, largest first.
pub fn polygons(mask: &[u8], w: usize, h: usize, tol: f64) -> Vec<Vec<i64>> {
    trace_outer(mask, w, h)
        .into_iter()
        .filter(|(_, a)| *a >= 2.0)
        .map(|(pts, _)| {
            simplify_closed(&pts, tol)
                .iter()
                .flat_map(|&(x, y)| [(x as f64 / w as f64 * 1000.0).round() as i64, (y as f64 / h as f64 * 1000.0).round() as i64])
                .collect()
        })
        .collect()
}
