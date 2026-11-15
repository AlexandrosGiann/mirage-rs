//! Mirage: describe an image as a structured record (scene, grounded objects
//! with boxes/masks, OCR text, relations, events, reasoning trace).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::cact::Cact;
use crate::decoder::Decoder;
use crate::image::{letterbox, Letterbox, Rgb};
use crate::masks::{polygons, rasterise, rle};
use crate::ops::linear;
use crate::reader::Reader;
use crate::record::{parse, Entity};
use crate::tokenizer::Tokenizer;
use crate::vision::{Encoded, VisionModel, CANVAS};
use crate::{err, Error, Result};

/// Description languages the released model was trained for.
pub const LANGUAGES: [&str; 7] = ["en", "de", "fr", "es", "it", "nl", "pl"];
pub const SECTIONS: [&str; 7] = ["scene", "objects", "regions", "text", "relations", "events", "reasoning"];
pub const FIELDS: [&str; 7] = ["bbox", "score", "point", "position", "mask", "description", "count"];
const DEFAULT_FIELDS: [&str; 5] = ["bbox", "score", "mask", "position", "description"];
const MASK_FORMATS: [&str; 4] = ["largest", "merged", "polygons", "bitmap"];
const N_OBJ: usize = 64;
const N_TXT: usize = 32;

fn sig(x: f32) -> f64 {
    1.0 / (1.0 + (-(x as f64)).exp())
}

/// Validated `describe` options (same names and defaults as the reference engine).
#[derive(Debug, Clone)]
pub struct Options {
    pub sections: Vec<String>,
    pub fields: Vec<String>,
    pub mask_format: String,
    pub conf: f64,
    pub text_conf: f64,
    pub iou: f64,
    pub nms: bool,
    pub agnostic_nms: bool,
    pub max_objects: usize,
    pub max_text_lines: usize,
    pub max_tokens: usize,
    pub mask_threshold: f64,
    pub mask_tolerance: f64,
    pub language: String,
}

impl Default for Options {
    fn default() -> Self {
        Options::from_json(&json!({})).unwrap()
    }
}

impl Options {
    /// Parse and validate a JSON options object.
    pub fn from_json(o: &Value) -> Result<Options> {
        let Some(o) = o.as_object() else { return err("options must be an object") };
        const KNOWN: [&str; 16] = ["sections", "fields", "mask_format", "conf", "text_conf", "iou", "nms", "agnostic_nms", "max_objects", "max_text_lines", "max_tokens", "mask_threshold", "mask_tolerance", "language", "form", "keyword_bias"];
        for k in o.keys() {
            if !KNOWN.contains(&k.as_str()) {
                return err(format!("option \"{k}\" has an unknown name"));
            }
        }
        if o.contains_key("form") {
            return err("forms are not supported: the reference engine's form prompt is not published");
        }
        let num = |k: &str, def: f64, lo: f64, hi: f64| -> Result<f64> {
            match o.get(k) {
                None | Some(Value::Null) => Ok(def),
                Some(v) => match v.as_f64() {
                    Some(x) if x.is_finite() && x >= lo && x <= hi => Ok(x),
                    _ => err(format!("{k} must be a number from {lo} to {hi}")),
                },
            }
        };
        let int = |k: &str, def: usize, lo: usize, hi: usize| -> Result<usize> {
            let v = num(k, def as f64, lo as f64, hi as f64)?;
            if v.fract() != 0.0 {
                return err(format!("{k} must be a whole number of at least {lo}"));
            }
            Ok(v as usize)
        };
        let boolean = |k: &str, def: bool| -> Result<bool> {
            match o.get(k) {
                None => Ok(def),
                Some(Value::Bool(b)) => Ok(*b),
                _ => err(format!("{k} must be true or false")),
            }
        };
        let names = |k: &str, allowed: &[&str], def: &[&str]| -> Result<Vec<String>> {
            match o.get(k) {
                None => Ok(def.iter().map(|s| s.to_string()).collect()),
                Some(Value::Array(a)) => {
                    let mut out: Vec<String> = Vec::new();
                    for n in a {
                        match n.as_str() {
                            Some(s) if allowed.contains(&s) => {
                                if !out.iter().any(|x| x == s) {
                                    out.push(s.to_string());
                                }
                            }
                            _ => return err(format!("{k} has an unknown name {n}")),
                        }
                    }
                    Ok(out)
                }
                _ => err(format!("{k} must be a list of names")),
            }
        };
        let language = match o.get("language") {
            None | Some(Value::Null) => "en".to_string(),
            Some(Value::String(s)) if LANGUAGES.contains(&s.as_str()) => s.clone(),
            _ => return err(format!("language must be one of {}", LANGUAGES.join(", "))),
        };
        let mask_format = match o.get("mask_format") {
            None | Some(Value::Null) => "largest".to_string(),
            Some(Value::String(s)) if MASK_FORMATS.contains(&s.as_str()) => s.clone(),
            _ => return err("mask_format must be largest, merged, polygons or bitmap"),
        };
        Ok(Options {
            sections: names("sections", &SECTIONS, &SECTIONS)?,
            fields: names("fields", &FIELDS, &DEFAULT_FIELDS)?,
            mask_format,
            conf: num("conf", 0.3, 0.0, 1.0)?,
            text_conf: num("text_conf", 0.3, 0.0, 1.0)?,
            iou: num("iou", 0.7, 0.0, 1.0)?,
            nms: boolean("nms", false)?,
            agnostic_nms: boolean("agnostic_nms", false)?,
            max_objects: int("max_objects", N_OBJ, 1, N_OBJ)?,
            max_text_lines: int("max_text_lines", N_TXT, 1, N_TXT)?,
            max_tokens: int("max_tokens", 768, 1, 1536)?,
            mask_threshold: num("mask_threshold", 0.5, 0.001, 0.999)?,
            mask_tolerance: num("mask_tolerance", 1.5, 0.0, 100.0)?,
            language,
        })
    }
}

/// A detector proposal.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub score: f64,
    /// centre cell on the 64x64 grid
    pub i: usize,
    pub j: usize,
    /// box on the canvas, normalised by the canvas size
    pub b: [f32; 4],
    /// box in original pixels
    pub px: [f32; 4],
    /// box on the public 0..1000 scale
    pub bbox: [i64; 4],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub perceive_ms: u64,
    pub decode_ms: u64,
    pub assemble_ms: u64,
}

/// `describe` output: the JSON record plus diagnostics.
pub struct Output {
    pub result: Value,
    /// raw decoder text (reference tokens dropped)
    pub raw: String,
    pub timing: Timing,
    /// (object, text) proposal counts
    pub proposals: (usize, usize),
}

struct Perception {
    lb: Letterbox,
    enc: Encoded,
    dm: Vec<f32>,
    objects: Vec<Proposal>,
    texts: Vec<Proposal>,
}

pub struct Mirage {
    cact: Cact,
    pub tokenizer: Tokenizer,
    vision: VisionModel,
    decoder: Decoder,
    reader: Reader,
    /// first reference token id (64 object then 32 text references follow)
    pub ref_base: u32,
}

fn iou_of(a: &[i64; 4], b: &[i64; 4]) -> f64 {
    let w = a[2].min(b[2]) - a[0].max(b[0]);
    let h = a[3].min(b[3]) - a[1].max(b[1]);
    if w <= 0 || h <= 0 {
        return 0.0;
    }
    let i = (w * h) as f64;
    i / (((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])) as f64 - i)
}

fn ms(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

impl Mirage {
    /// Load from `mirage.cact` bytes.
    pub fn new(bytes: Vec<u8>) -> Result<Mirage> {
        let cact = Cact::new(bytes)?;
        let tokenizer = Tokenizer::new(cact.bytes("tokenizer")?)?;
        let vision = VisionModel::new(&cact)?;
        let decoder = Decoder::new(&cact)?;
        let reader = Reader::new(&cact)?;
        let ref_base = tokenizer.size() as u32;
        if decoder.v != tokenizer.size() + N_OBJ + N_TXT {
            return err("unexpected vocabulary layout");
        }
        Ok(Mirage { cact, tokenizer, vision, decoder, reader, ref_base })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Mirage> {
        let p = path.as_ref();
        let bytes = std::fs::read(p).map_err(|e| Error(format!("cannot read {}: {e}", p.display())))?;
        Mirage::new(bytes)
    }

    fn proposals(det: &[f32], kind: usize, conf: f64, max_n: usize, vh8: usize, vw8: usize, lb: &Letterbox, img: &Rgb) -> Vec<Proposal> {
        let at = |i: usize, j: usize| det[(i * 64 + j) * 10 + kind];
        let mut peaks = Vec::new();
        for i in 0..vh8 {
            for j in 0..vw8 {
                let v = at(i, j);
                if sig(v) < conf {
                    continue;
                }
                let mut ok = true;
                'n: for di in -1i64..=1 {
                    for dj in -1i64..=1 {
                        let (a, b) = (i as i64 + di, j as i64 + dj);
                        if (di != 0 || dj != 0) && a >= 0 && b >= 0 && (a as usize) < vh8 && (b as usize) < vw8 && at(a as usize, b as usize) > v {
                            ok = false;
                            break 'n;
                        }
                    }
                }
                if ok {
                    peaks.push((sig(v), i, j));
                }
            }
        }
        peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::new();
        let (wd, hd) = (img.width as f64, img.height as f64);
        for (score, i, j) in peaks {
            if out.len() >= max_n {
                break;
            }
            let o = (i * 64 + j) * 10;
            let w = det[o + 2 + kind * 2].abs() as f64;
            let h = det[o + 3 + kind * 2].abs() as f64;
            let xc = j as f64 + det[o + 6 + kind * 2] as f64;
            let yc = i as f64 + det[o + 7 + kind * 2] as f64;
            let cw = lb.w as f64 / CANVAS as f64;
            let ch = lb.h as f64 / CANVAS as f64;
            let b = [((xc - w / 2.0) / 64.0).max(0.0), ((yc - h / 2.0) / 64.0).max(0.0), ((xc + w / 2.0) / 64.0).min(cw), ((yc + h / 2.0) / 64.0).min(ch)];
            if b[2] <= b[0] || b[3] <= b[1] {
                continue;
            }
            let c = CANVAS as f64;
            let (sx, sy) = (lb.sx as f64, lb.sy as f64);
            let px = [b[0] * c / sx, b[1] * c / sy, (b[2] * c / sx).min(wd), (b[3] * c / sy).min(hd)];
            let bbox = [(px[0] / wd * 1000.0).round() as i64, (px[1] / hd * 1000.0).round() as i64, (px[2] / wd * 1000.0).round() as i64, (px[3] / hd * 1000.0).round() as i64];
            out.push(Proposal { score, i, j, b: b.map(|v| v as f32), px: px.map(|v| v as f32), bbox });
        }
        // slots and reference tokens follow the raster order of the centre cells
        out.sort_by_key(|p| (p.i, p.j));
        out
    }

    fn perceive(&self, img: &Rgb, o: &Options) -> Perception {
        let lb = letterbox(img, CANVAS);
        let enc = self.vision.encode(&lb.canvas, lb.h, lb.w);
        let dm = self.vision.neck_forward(&enc);
        let det = self.vision.detect(&dm);
        let (vh8, vw8) = (lb.h.div_ceil(8), lb.w.div_ceil(8));
        let objects = Self::proposals(&det, 0, o.conf, o.max_objects, vh8, vw8, &lb, img);
        let texts = Self::proposals(&det, 1, o.text_conf, o.max_text_lines, vh8, vw8, &lb, img);
        Perception { lb, enc, dm, objects, texts }
    }

    /// Detector proposals only (objects, text lines), without the decoder.
    pub fn detect(&self, img: &Rgb, o: &Options) -> (Vec<Proposal>, Vec<Proposal>) {
        let p = self.perceive(img, o);
        (p.objects, p.texts)
    }

    fn generate(&self, st: &Perception, prefix: &[u32], max_tokens: usize, on_token: &mut dyn FnMut(u32)) -> Result<Vec<u32>> {
        // image memory: 256 cells then one slot per proposal
        let mut slots: Vec<(&Proposal, u32)> = Vec::new();
        for (k, p) in st.objects.iter().enumerate() {
            slots.push((p, self.ref_base + k as u32));
        }
        for (k, p) in st.texts.iter().enumerate() {
            slots.push((p, self.ref_base + (N_OBJ + k) as u32));
        }
        let mn = 256 + slots.len();
        let mut m = vec![0f32; mn * 512];
        let mut valid = vec![false; mn];
        m[..256 * 512].copy_from_slice(&st.enc.e);
        for (p, v) in valid.iter_mut().enumerate().take(256) {
            *v = (p >> 4) < st.enc.gh && (p & 15) < st.enc.gw;
        }
        for (k, (p, r)) in slots.iter().enumerate() {
            let s = self.vision.slot_vector(&st.dm, &st.enc, p.b, p.i, p.j, self.decoder.embedding(*r as usize));
            m[(256 + k) * 512..(257 + k) * 512].copy_from_slice(&s);
            valid[256 + k] = true;
        }
        let v = self.decoder.v;
        let mut allowed = vec![true; v];
        let first_tag = self.tokenizer.id("<|en|>")? as usize;
        for a in allowed.iter_mut().take(v).skip(first_tag) {
            *a = false; // language tags and every reference token
        }
        for (_, r) in &slots {
            allowed[*r as usize] = true;
        }
        for id in [self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.unk_id] {
            if (id as usize) < v {
                allowed[id as usize] = false;
            }
        }
        let mut ds = self.decoder.start(&m, mn, valid);
        let mut h = Vec::new();
        for &t in prefix {
            h = self.decoder.step(&self.cact, &mut ds, t);
        }
        let mut ids = Vec::new();
        let mut lg = vec![0f32; v];
        for _ in 0..max_tokens {
            self.decoder.logits(&h, &mut lg);
            let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &x) in lg.iter().enumerate() {
                if allowed[i] && x > bv {
                    bv = x;
                    best = i;
                }
            }
            if best as u32 == self.tokenizer.eos_id {
                break;
            }
            ids.push(best as u32);
            on_token(best as u32);
            h = self.decoder.step(&self.cact, &mut ds, best as u32);
        }
        Ok(ids)
    }

    fn masks(&self, st: &Perception, img: &Rgb, wanted: &[&Proposal], o: &Options) -> Vec<Value> {
        if wanted.is_empty() {
            return Vec::new();
        }
        let c = self.vision.coefficients(&st.dm);
        let p = self.vision.prototypes(&st.dm, &st.enc);
        let bitmap = o.mask_format == "bitmap";
        // polygon raster: original size capped at 1024 on the long side
        let scale = if bitmap { 1.0 } else { (1024.0 / img.width.max(img.height) as f32).min(1.0) };
        let w = ((img.width as f32 * scale).round() as usize).max(1);
        let h = ((img.height as f32 * scale).round() as usize).max(1);
        wanted
            .iter()
            .map(|q| {
                let cf = &c[(q.i * 64 + q.j) * 32..(q.i * 64 + q.j + 1) * 32];
                let logits = linear(&p, 128 * 128, cf, 1, 32, None);
                let bx = q.px.map(|v| v * scale);
                let m = rasterise(&logits, st.lb.sx / scale, st.lb.sy / scale, w, h, bx, o.mask_threshold as f32);
                if bitmap {
                    let (size, counts) = rle(&m, w, h);
                    return json!({ "size": size, "counts": counts });
                }
                let polys = polygons(&m, w, h, o.mask_tolerance);
                match o.mask_format.as_str() {
                    "polygons" => json!(polys),
                    "merged" => json!(polys.concat()),
                    _ => json!(polys.into_iter().next().unwrap_or_default()),
                }
            })
            .collect()
    }

    /// Run the full model. `on_token` sees every generated token id.
    pub fn describe(&self, img: &Rgb, o: &Options, on_token: &mut dyn FnMut(u32)) -> Result<Output> {
        let t0 = Instant::now();
        let st = self.perceive(img, o);
        let perceive_ms = ms(t0);
        let t1 = Instant::now();
        let sec: HashSet<&str> = o.sections.iter().map(String::as_str).collect();
        let fld: HashSet<&str> = o.fields.iter().map(String::as_str).collect();
        let need_decoder = ["scene", "objects", "regions", "relations", "events", "reasoning"].iter().any(|s| sec.contains(s));
        let mut rec = Default::default();
        let mut raw = String::new();
        if need_decoder {
            let head = self.tokenizer.encode(&format!("<think>lang: {}", o.language));
            let mut prefix = vec![self.tokenizer.id(&format!("<|{}|>", o.language))?];
            prefix.extend_from_slice(&head);
            let gen = self.generate(&st, &prefix, o.max_tokens, on_token)?;
            let ids: Vec<u32> = head.iter().chain(&gen).copied().collect();
            let text_ids: Vec<u32> = ids.iter().copied().filter(|&i| i < self.ref_base).collect();
            raw = self.tokenizer.decode(&text_ids);
            rec = parse(&ids, &self.tokenizer, self.ref_base, N_OBJ);
        }
        let rec: crate::record::Record = rec;
        let decode_ms = ms(t1);
        let t2 = Instant::now();
        let mut result = Map::new();
        if sec.contains("scene") {
            result.insert("scene".into(), Value::Object(rec.scene.clone()));
        }
        // objects: one entry per referenced object slot
        let mut objects: Vec<(&Entity, &Proposal)> = Vec::new();
        let mut used = HashSet::new();
        for e in &rec.entities {
            for &r in &e.refs {
                if r >= N_OBJ {
                    continue;
                }
                if let Some(p) = st.objects.get(r) {
                    if used.insert(r) {
                        objects.push((e, p));
                    }
                }
            }
        }
        if o.nms {
            let mut sorted = objects.clone();
            sorted.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));
            let mut kept: Vec<(&Entity, &Proposal)> = Vec::new();
            for it in sorted {
                if kept.iter().any(|k| (o.agnostic_nms || k.0.name == it.0.name) && iou_of(&k.1.bbox, &it.1.bbox) > o.iou) {
                    continue;
                }
                kept.push(it);
            }
            // keep the decoder's order
            objects.retain(|x| kept.iter().any(|k| std::ptr::eq(k.1, x.1)));
        }
        let describe = |e: &Entity, r: &mut Map<String, Value>| {
            if fld.contains("count") {
                if let Some(c) = e.count {
                    r.insert("count".into(), json!(c));
                }
            }
            if fld.contains("position") && !e.position.is_empty() {
                r.insert("position".into(), json!(e.position));
            }
            if fld.contains("description") && !e.description.is_empty() {
                r.insert("description".into(), Value::Object(e.description.clone()));
            }
        };
        let point = |b: &[i64; 4]| json!([((b[0] + b[2]) as f64 / 2.0).round() as i64, ((b[1] + b[3]) as f64 / 2.0).round() as i64]);
        let score = |s: f64| json!((s * 1000.0).round() / 1000.0);
        if sec.contains("objects") {
            let masks = if fld.contains("mask") { self.masks(&st, img, &objects.iter().map(|x| x.1).collect::<Vec<_>>(), o) } else { Vec::new() };
            let list: Vec<Value> = objects
                .iter()
                .enumerate()
                .map(|(k, (e, p))| {
                    let mut r = Map::new();
                    r.insert("id".into(), json!(e.id));
                    r.insert("name".into(), json!(e.name));
                    if fld.contains("bbox") {
                        r.insert("bbox".into(), json!(p.bbox));
                    }
                    if fld.contains("score") {
                        r.insert("score".into(), score(p.score));
                    }
                    if fld.contains("point") {
                        r.insert("point".into(), point(&p.bbox));
                    }
                    if fld.contains("mask") {
                        r.insert("mask".into(), masks[k].clone());
                    }
                    describe(e, &mut r);
                    Value::Object(r)
                })
                .collect();
            result.insert("objects".into(), Value::Array(list));
        }
        if sec.contains("regions") {
            let list: Vec<Value> = rec
                .entities
                .iter()
                .filter(|e| e.refs.is_empty())
                .map(|e| {
                    let mut r = Map::new();
                    r.insert("id".into(), json!(e.id));
                    r.insert("name".into(), json!(e.name));
                    describe(e, &mut r);
                    Value::Object(r)
                })
                .collect();
            result.insert("regions".into(), Value::Array(list));
        }
        if sec.contains("text") {
            let mut on: HashMap<usize, i64> = HashMap::new();
            for e in &rec.entities {
                for &r in &e.refs {
                    if r >= N_OBJ {
                        on.insert(r - N_OBJ, e.id);
                    }
                }
            }
            for &(k, ent) in &rec.texts {
                if let Some(ent) = ent {
                    on.insert(k, ent);
                }
            }
            let mut lines = Vec::new();
            for (k, p) in st.texts.iter().enumerate() {
                let (string, _) = self.reader.read(img, p.px);
                if string.trim().is_empty() {
                    continue;
                }
                let mut r = Map::new();
                r.insert("string".into(), json!(string));
                if fld.contains("bbox") {
                    r.insert("bbox".into(), json!(p.bbox));
                }
                if fld.contains("score") {
                    r.insert("score".into(), score(p.score));
                }
                if fld.contains("point") {
                    r.insert("point".into(), point(&p.bbox));
                }
                if let Some(e) = on.get(&k) {
                    r.insert("on".into(), json!(e));
                }
                lines.push(Value::Object(r));
            }
            result.insert("text".into(), Value::Array(lines));
        }
        if sec.contains("relations") {
            result.insert("relations".into(), Value::Array(rec.relations.clone()));
        }
        if sec.contains("events") {
            result.insert("events".into(), Value::Array(rec.events.clone()));
        }
        if sec.contains("reasoning") {
            result.insert("reasoning".into(), json!(rec.reasoning));
        }
        Ok(Output {
            result: Value::Object(result),
            raw,
            timing: Timing { perceive_ms, decode_ms, assemble_ms: ms(t2) },
            proposals: (st.objects.len(), st.texts.len()),
        })
    }
}
