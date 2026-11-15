//! Unit tests always run; model tests run when mirage.cact is found
//! (MIRAGE_WEIGHTS, ./mirage.cact or ./models/mirage.cact).

use mirage::serde_json::{json, Value};
use mirage::{cact::fwht, image, masks, record, Mirage, Options};

#[test]
fn fwht_matches_the_explicit_hadamard_product() {
    let n: usize = 16;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).sin() + i as f32 * 0.1).collect();
    let mut y = x.clone();
    fwht(&mut y);
    for r in 0..n {
        let s: f32 = (0..n).map(|c| if (r & c).count_ones() % 2 == 1 { -x[c] } else { x[c] }).sum();
        assert!((s - y[r]).abs() < 1e-4, "row {r}");
    }
}

#[test]
fn letterbox_rounds_each_side_up_to_a_multiple_of_32() {
    for (w, h, ew, eh) in [(342, 512, 352, 512), (640, 488, 512, 416), (256, 256, 512, 512), (1, 1, 512, 512), (4000, 10, 512, 32), (10, 4000, 32, 512)] {
        let img = image::Rgb::new(w, h, vec![0; w * h * 3]).unwrap();
        let lb = image::letterbox(&img, 512);
        assert_eq!((lb.w, lb.h), (ew, eh), "{w}x{h}");
        assert_eq!(lb.canvas.len(), 512 * 512 * 3);
    }
}

#[test]
fn png_decode_of_the_sample() {
    let img = image::decode(include_bytes!("data/sample.png")).unwrap();
    assert_eq!((img.width, img.height), (480, 360));
    assert!(image::decode(b"not an image").is_err());
}

#[test]
fn coco_rle_is_column_major_and_sums_to_the_area() {
    let (w, h) = (4, 3);
    let mut m = vec![0u8; w * h];
    m[w + 2] = 1;
    m[2 * w + 2] = 1;
    let (size, counts) = masks::rle(&m, w, h);
    assert_eq!(size, [3, 4]);
    assert_eq!(counts, vec![7, 2, 3]);
    assert_eq!(counts.iter().sum::<usize>(), w * h);
}

#[test]
fn polygons_trace_a_filled_square_outer_boundary_only() {
    let (w, h) = (20, 10);
    let mut m = vec![0u8; w * h];
    for y in 2..8 {
        for x in 5..15 {
            m[y * w + x] = 1;
        }
    }
    m[5 * w + 10] = 0; // a hole is ignored
    let p = masks::polygons(&m, w, h, 0.5);
    assert_eq!(p.len(), 1);
    let xs: Vec<i64> = p[0].iter().step_by(2).copied().collect();
    let ys: Vec<i64> = p[0].iter().skip(1).step_by(2).copied().collect();
    assert_eq!([*xs.iter().min().unwrap(), *xs.iter().max().unwrap(), *ys.iter().min().unwrap(), *ys.iter().max().unwrap()], [250, 750, 200, 800]);
    assert!(masks::polygons(&vec![0u8; w * h], w, h, 1.5).is_empty());
}

#[test]
fn record_parser_reads_trace_scene_entities_relations_events() {
    let pieces = ["<think>lang: en | a | b</think>S background=sky; lighting=day\nE1 stop sign @center *", "R0", " {color=red; shape=octagonal}\nE2 tree #3 @background *", "R1", "R2", " {state=bare}\nE3 building @left {color=gray}\nR 1 attached_to 2\nV 1 waves 2\nT ", "R65", " on 3\n"];
    let mut strs: Vec<&str> = Vec::new();
    let mut ids = Vec::new();
    for p in pieces {
        if let Some(n) = p.strip_prefix('R').filter(|n| n.chars().all(|c| c.is_ascii_digit())) {
            ids.push(1000 + n.parse::<u32>().unwrap());
        } else {
            strs.push(p);
            ids.push(strs.len() as u32 - 1);
        }
    }
    let dec = |xs: &[u32]| xs.iter().map(|&i| strs[i as usize]).collect::<String>();
    let r = record::parse_with(&ids, &dec, 1000, 64);
    assert_eq!(r.reasoning, "lang: en | a | b");
    assert_eq!(Value::Object(r.scene.clone()), json!({ "background": "sky", "lighting": "day" }));
    let ents: Vec<_> = r.entities.iter().map(|e| (e.id, e.name.as_str(), e.position.as_str(), e.refs.clone(), e.count)).collect();
    assert_eq!(ents, vec![(1, "stop sign", "center", vec![0], None), (2, "tree", "background", vec![1, 2], Some(3)), (3, "building", "left", vec![], None)]);
    assert_eq!(Value::Object(r.entities[0].description.clone()), json!({ "color": "red", "shape": "octagonal" }));
    assert_eq!(r.relations, vec![json!({ "subject": 1, "predicate": "attached_to", "object": 2 })]);
    assert_eq!(r.events, vec![json!({ "actor": 1, "action": "waves", "target": 2 })]);
    assert_eq!(r.texts, vec![(1, Some(3))]);
}

#[test]
fn option_validation_mirrors_the_reference_engine() {
    assert_eq!(Options::default().conf, 0.3);
    let e = |v: Value| Options::from_json(&v).unwrap_err().to_string();
    assert!(e(json!({ "mask_format": "svg" })).contains("largest, merged, polygons or bitmap"));
    assert!(e(json!({ "fields": ["bbox", "colour"] })).contains("unknown name"));
    assert!(e(json!({ "conf": 2 })).contains("number from 0 to 1"));
    assert!(e(json!({ "max_objects": 1.5 })).contains("whole number"));
    assert!(e(json!({ "nope": 1 })).contains("unknown name"));
    assert!(e(json!({ "language": "xx" })).contains("language"));
    assert!(e(json!([])).contains("object"));
}

fn model() -> Option<Mirage> {
    match mirage::find_weights() {
        Some(p) => Some(Mirage::open(p).expect("weights load")),
        None => {
            eprintln!("skipping model test: mirage.cact not found (set MIRAGE_WEIGHTS)");
            None
        }
    }
}

#[test]
fn end_to_end_on_the_sample() {
    let Some(m) = model() else { return };
    let img = image::decode(include_bytes!("data/sample.png")).unwrap();
    // proposals measured with the official libcactus 4.1.1 on the same PNG
    let (objects, _) = m.detect(&img, &Options::default());
    for (bx, score) in [([407, 173, 599, 427], 0.612), ([425, 177, 579, 1000], 0.669), ([83, 558, 300, 791], 0.705)] {
        let p = objects.iter().find(|o| o.bbox.iter().zip(bx).all(|(a, b)| (a - b).abs() <= 2)).unwrap_or_else(|| panic!("proposal {bx:?} missing"));
        assert!((p.score - score).abs() < 0.01, "score {} vs {score}", p.score);
    }
    // same record as the mirage-js port on this image
    let o = Options::from_json(&json!({ "fields": ["bbox", "score", "position", "description"], "sections": ["scene", "objects", "text", "relations", "reasoning"] })).unwrap();
    let out = m.describe(&img, &o, &mut |_| {}).unwrap();
    let expected: Value = mirage::serde_json::from_str(include_str!("data/sample.expected.json")).unwrap();
    let names = |v: &Value| v["objects"].as_array().unwrap().iter().map(|o| (o["name"].clone(), o["bbox"].clone())).collect::<Vec<_>>();
    assert_eq!(names(&out.result), names(&expected));
    assert_eq!(out.result["relations"], expected["relations"]);
    assert!(out.result["reasoning"].as_str().unwrap().starts_with("lang: en |"));
    // masks in every format, and a non-ASCII language
    for fmt in ["largest", "polygons", "merged", "bitmap"] {
        let o = Options::from_json(&json!({ "mask_format": fmt, "sections": ["objects"], "max_tokens": 120 })).unwrap();
        let r = m.describe(&img, &o, &mut |_| {}).unwrap().result;
        let objs = r["objects"].as_array().unwrap();
        assert!(!objs.is_empty(), "{fmt}");
        for ob in objs {
            assert!(!ob["mask"].is_null(), "{fmt}");
        }
    }
    let de = Options::from_json(&json!({ "language": "de", "sections": ["reasoning"], "max_tokens": 40 })).unwrap();
    let r = m.describe(&img, &de, &mut |_| {}).unwrap().result;
    assert!(r["reasoning"].as_str().unwrap().starts_with("lang: de"));
}
