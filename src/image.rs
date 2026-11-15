//! Image helpers: decoding (PNG/JPEG through pure-Rust decoders, EXIF
//! orientation applied), letterboxing onto Mirage's 512 px canvas and the
//! crop/resize used by the reader.

use crate::{err, Result};

/// Packed 8-bit RGB image.
#[derive(Clone, Debug)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Rgb> {
        if width == 0 || height == 0 || data.len() != width * height * 3 {
            return err("image must be width x height x 3 packed RGB bytes");
        }
        Ok(Rgb { width, height, data })
    }
}

/// Decode PNG or JPEG bytes into packed RGB (alpha dropped), applying EXIF orientation.
pub fn decode(bytes: &[u8]) -> Result<Rgb> {
    use image::{ImageDecoder, ImageReader};
    let fmt = if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        image::ImageFormat::Png
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        image::ImageFormat::Jpeg
    } else {
        return err("unsupported image format: pass PNG or JPEG bytes");
    };
    let mut reader = ImageReader::with_format(std::io::Cursor::new(bytes), fmt);
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(1 << 30);
    reader.limits(limits);
    let mut dec = reader.into_decoder().map_err(|e| crate::Error(format!("could not decode image: {e}")))?;
    let orientation = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(dec).map_err(|e| crate::Error(format!("could not decode image: {e}")))?;
    img.apply_orientation(orientation);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Rgb::new(w, h, rgb.into_raw())
}

/// Bilinear resize (half-pixel centres) of region [x0,y0,x1,y1) to w x h; float RGB in pixel units.
pub fn resize_region(img: &Rgb, x0: f32, y0: f32, x1: f32, y1: f32, w: usize, h: usize) -> Vec<f32> {
    let (ww, hh, d) = (img.width, img.height, &img.data);
    let mut out = vec![0f32; w * h * 3];
    let sx = (x1 - x0) as f64 / w as f64;
    let sy = (y1 - y0) as f64 / h as f64;
    for y in 0..h {
        let fy = (y0 as f64 + (y as f64 + 0.5) * sy - 0.5).clamp(0.0, (hh - 1) as f64);
        let iy = fy.floor() as usize;
        let ty = fy - iy as f64;
        let iy1 = (iy + 1).min(hh - 1);
        for x in 0..w {
            let fx = (x0 as f64 + (x as f64 + 0.5) * sx - 0.5).clamp(0.0, (ww - 1) as f64);
            let ix = fx.floor() as usize;
            let tx = fx - ix as f64;
            let ix1 = (ix + 1).min(ww - 1);
            let (a, b, c, e) = ((iy * ww + ix) * 3, (iy * ww + ix1) * 3, (iy1 * ww + ix) * 3, (iy1 * ww + ix1) * 3);
            for k in 0..3 {
                let v = (d[a + k] as f64 * (1.0 - tx) + d[b + k] as f64 * tx) * (1.0 - ty) + (d[c + k] as f64 * (1.0 - tx) + d[e + k] as f64 * tx) * ty;
                out[(y * w + x) * 3 + k] = v as f32;
            }
        }
    }
    out
}

/// Area-average resize for shrinking (bilinear when enlarging either axis).
pub fn resize_area(img: &Rgb, w: usize, h: usize) -> Vec<f32> {
    let (ww, hh, d) = (img.width, img.height, &img.data);
    if w >= ww || h >= hh {
        return resize_region(img, 0.0, 0.0, ww as f32, hh as f32, w, h);
    }
    let mut out = vec![0f32; w * h * 3];
    let sx = ww as f64 / w as f64;
    let sy = hh as f64 / h as f64;
    for y in 0..h {
        let ya = y as f64 * sy;
        let yb = ya + sy;
        for x in 0..w {
            let xa = x as f64 * sx;
            let xb = xa + sx;
            let (mut r, mut g, mut b, mut wt) = (0f64, 0f64, 0f64, 0f64);
            for yy in ya.floor() as usize..hh.min(yb.ceil() as usize) {
                let wy = yb.min(yy as f64 + 1.0) - ya.max(yy as f64);
                for xx in xa.floor() as usize..ww.min(xb.ceil() as usize) {
                    let wx = xb.min(xx as f64 + 1.0) - xa.max(xx as f64);
                    let q = wx * wy;
                    let p = (yy * ww + xx) * 3;
                    r += d[p] as f64 * q;
                    g += d[p + 1] as f64 * q;
                    b += d[p + 2] as f64 * q;
                    wt += q;
                }
            }
            let o = (y * w + x) * 3;
            out[o] = (r / wt) as f32;
            out[o + 1] = (g / wt) as f32;
            out[o + 2] = (b / wt) as f32;
        }
    }
    out
}

/// Image placed on the model canvas.
pub struct Letterbox {
    /// canvas x canvas x 3, normalised to [-1, 1]; zero outside the image
    pub canvas: Vec<f32>,
    pub w: usize,
    pub h: usize,
    /// canvas px per original px
    pub sx: f32,
    pub sy: f32,
}

/// Fit at the top-left: the long side becomes `canvas` and each side is rounded
/// up to a multiple of 32 (a slight stretch, matching the reference engine).
pub fn letterbox(img: &Rgb, canvas: usize) -> Letterbox {
    let r = canvas as f64 / img.width.max(img.height) as f64;
    let q = 32;
    let fit = |v: usize| -> usize {
        let s = ((v as f64 * r).round() as usize).max(1);
        (s.div_ceil(q) * q).clamp(q, canvas)
    };
    let (w, h) = (fit(img.width), fit(img.height));
    let px = resize_area(img, w, h);
    let mut out = vec![0f32; canvas * canvas * 3];
    for y in 0..h {
        for x in 0..w {
            let (s, d) = ((y * w + x) * 3, (y * canvas + x) * 3);
            for k in 0..3 {
                out[d + k] = px[s + k] / 127.5 - 1.0;
            }
        }
    }
    Letterbox { canvas: out, w, h, sx: w as f32 / img.width as f32, sy: h as f32 / img.height as f32 }
}
