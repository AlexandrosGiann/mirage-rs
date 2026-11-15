//! SentencePiece BPE tokenizer read from the `.cact` RAW attachment.

use std::collections::HashMap;

use crate::{err, Result};

const META: char = '\u{2581}';
const UNKNOWN: u8 = 1;
const CONTROL: u8 = 2;
const USER: u8 = 3;
const BYTE: u8 = 4;

pub struct Tokenizer {
    pub pad_id: u32,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    add_dummy: bool,
    byte_fallback: bool,
    pieces: Vec<String>,
    scores: Vec<f32>,
    types: Vec<u8>,
    p2id: HashMap<String, u32>,
    byte_id: HashMap<u8, u32>,
    markers: Vec<String>,
}

fn byte_of(piece: &str) -> Option<u8> {
    // "<0xAB>"
    piece.get(3..5).and_then(|h| u8::from_str_radix(h, 16).ok())
}

impl Tokenizer {
    pub fn new(blob: &[u8]) -> Result<Tokenizer> {
        let u32_at = |o: usize| -> Result<u32> {
            blob.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| crate::Error("tokenizer blob truncated".into()))
        };
        if blob.len() < 24 {
            return err("tokenizer blob truncated");
        }
        let n = u32_at(0)? as usize;
        let mut t = Tokenizer {
            pad_id: u32_at(4)?,
            eos_id: u32_at(8)?,
            bos_id: u32_at(12)?,
            unk_id: u32_at(16)?,
            add_dummy: blob[20] == 1,
            byte_fallback: blob[21] == 1,
            pieces: Vec::with_capacity(n),
            scores: Vec::with_capacity(n),
            types: Vec::with_capacity(n),
            p2id: HashMap::with_capacity(n),
            byte_id: HashMap::new(),
            markers: Vec::new(),
        };
        let mut o = 24;
        for i in 0..n {
            if o + 7 > blob.len() {
                return err("tokenizer blob truncated");
            }
            t.scores.push(f32::from_le_bytes(blob[o..o + 4].try_into().unwrap()));
            let ty = blob[o + 4];
            t.types.push(ty);
            let len = u16::from_le_bytes([blob[o + 5], blob[o + 6]]) as usize;
            o += 7;
            if o + len > blob.len() {
                return err("tokenizer blob truncated");
            }
            let p = String::from_utf8_lossy(&blob[o..o + len]).into_owned();
            o += len;
            if ty == BYTE {
                if let Some(b) = byte_of(&p) {
                    t.byte_id.insert(b, i as u32);
                }
            }
            if ty == USER {
                t.markers.push(p.clone());
            }
            t.p2id.insert(p.clone(), i as u32);
            t.pieces.push(p);
        }
        t.markers.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()));
        Ok(t)
    }

    pub fn size(&self) -> usize {
        self.pieces.len()
    }

    pub fn id(&self, piece: &str) -> Result<u32> {
        match self.p2id.get(piece) {
            Some(&i) => Ok(i),
            None => err(format!("unknown piece {piece}")),
        }
    }

    fn bpe(&self, seg: &str, ids: &mut Vec<u32>) {
        let mut syms: Vec<String> = seg.chars().map(String::from).collect();
        while syms.len() > 1 {
            let (mut best, mut bj) = (f32::NEG_INFINITY, None);
            for j in 0..syms.len() - 1 {
                let m = format!("{}{}", syms[j], syms[j + 1]);
                if let Some(&id) = self.p2id.get(&m) {
                    if self.scores[id as usize] > best {
                        best = self.scores[id as usize];
                        bj = Some(j);
                    }
                }
            }
            let Some(j) = bj else { break };
            let next = syms.remove(j + 1);
            syms[j].push_str(&next);
        }
        for s in syms {
            if let Some(&id) = self.p2id.get(&s) {
                ids.push(id);
            } else if self.byte_fallback {
                ids.extend(s.bytes().filter_map(|b| self.byte_id.get(&b).copied()));
            } else {
                ids.push(self.unk_id);
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if text.is_empty() {
            return ids;
        }
        let mut esc = text.replace(' ', &META.to_string());
        if self.add_dummy {
            esc.insert(0, META);
        }
        let mut buf = String::new();
        let mut i = 0;
        while i < esc.len() {
            if let Some(m) = self.markers.iter().find(|m| esc[i..].starts_with(m.as_str())) {
                if !buf.is_empty() {
                    self.bpe(&buf, &mut ids);
                    buf.clear();
                }
                ids.push(self.p2id[m]);
                i += m.len();
            } else {
                let c = esc[i..].chars().next().unwrap();
                buf.push(c);
                i += c.len_utf8();
            }
        }
        if !buf.is_empty() {
            self.bpe(&buf, &mut ids);
        }
        ids
    }

    /// Raw UTF-8 bytes of a token run (META kept), for incremental decoding.
    fn bytes_of(&self, ids: &[u32], out: &mut Vec<u8>) {
        for &i in ids {
            let i = i as usize;
            if i >= self.pieces.len() {
                continue;
            }
            match self.types[i] {
                BYTE => out.extend(byte_of(&self.pieces[i])),
                CONTROL | UNKNOWN => {}
                _ => out.extend_from_slice(self.pieces[i].as_bytes()),
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        self.bytes_of(ids, &mut b);
        let s = String::from_utf8_lossy(&b).replace(META, " ");
        if self.add_dummy {
            if let Some(r) = s.strip_prefix(' ') {
                return r.to_string();
            }
        }
        s
    }
}
