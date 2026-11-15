//! Parse Mirage's compact line output into a structured record.
//!
//! ```text
//! <think>trace</think>
//! S key=value; key=value                          scene fields
//! E<n> name [#count] @position *<ref>... {k=v; ...}  entity (refs ground it)
//! R <a> predicate <b>                             relation between entity ids
//! T <ref> [on] <entity>                           text line placement
//! V <actor> action [<target>]                     event (other single-letter lines)
//! ```
//!
//! Reference tokens become `\u{1}<index>\u{2}` markers before parsing.

use serde_json::{Map, Value};

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Default)]
pub struct Entity {
    pub id: i64,
    pub name: String,
    pub position: String,
    pub refs: Vec<usize>,
    pub description: Map<String, Value>,
    pub count: Option<i64>,
}

#[derive(Debug, Default)]
pub struct Record {
    pub scene: Map<String, Value>,
    pub entities: Vec<Entity>,
    pub relations: Vec<Value>,
    pub events: Vec<Value>,
    /// (text slot index, entity id it sits on)
    pub texts: Vec<(usize, Option<i64>)>,
    pub reasoning: String,
}

fn kv(s: &str) -> Map<String, Value> {
    let mut out = Map::new();
    for part in s.split(';') {
        if let Some(i) = part.find('=') {
            let k = part[..i].trim();
            if !k.is_empty() {
                out.insert(k.to_string(), Value::String(part[i + 1..].trim().to_string()));
            }
        }
    }
    out
}

/// Entity id: "E3" / "3" -> 3, anything else stays a string.
fn id_of(s: &str) -> Value {
    let t = s.strip_prefix('E').unwrap_or(s);
    match t.parse::<i64>() {
        Ok(n) => Value::from(n),
        Err(_) => Value::String(s.to_string()),
    }
}

/// Text after `tag` when it is followed by whitespace.
fn after<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let r = line.strip_prefix(tag)?;
    r.starts_with(char::is_whitespace).then(|| r.trim_start())
}

/// Pulls `\u{1}n\u{2}` markers out of `s`, replacing each with a space.
fn take_refs(s: &str) -> (Vec<usize>, String) {
    let mut refs = Vec::new();
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(a) = rest.find('\u{1}') {
        out.push_str(&rest[..a]);
        let tail = &rest[a + 1..];
        match tail.find('\u{2}') {
            Some(b) if b > 0 && tail[..b].bytes().all(|c| c.is_ascii_digit()) => {
                refs.push(tail[..b].parse().unwrap_or(0));
                out.push(' ');
                rest = &tail[b + 1..];
            }
            _ => {
                out.push('\u{1}');
                rest = tail;
            }
        }
    }
    out.push_str(rest);
    (refs, out)
}

fn entity(id: i64, rest: &str) -> Entity {
    let mut e = Entity { id, ..Default::default() };
    let mut rest = rest.to_string();
    if let Some(b) = rest.find('{') {
        let d = rest[b + 1..].trim_end();
        e.description = kv(d.strip_suffix('}').unwrap_or(d));
        rest.truncate(b);
    }
    let (refs, mut rest) = take_refs(&rest);
    e.refs = refs;
    if let Some(s) = rest.find('*') {
        rest.truncate(s);
    }
    if let Some(a) = rest.find('@') {
        e.position = rest[a + 1..].trim().to_string();
        rest.truncate(a);
    }
    // "#<n>" count (first occurrence)
    let mut search = 0;
    while let Some(h) = rest[search..].find('#').map(|h| h + search) {
        let after_hash = &rest[h + 1..];
        let digits_at = after_hash.len() - after_hash.trim_start().len();
        let digits: String = after_hash[digits_at..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            e.count = digits.parse().ok();
            rest.replace_range(h..h + 1 + digits_at + digits.len(), "");
            break;
        }
        search = h + 1;
    }
    e.name = rest.trim().to_string();
    e
}

pub fn parse(ids: &[u32], tok: &Tokenizer, ref_base: u32, n_obj: usize) -> Record {
    parse_with(ids, &|r: &[u32]| tok.decode(r), ref_base, n_obj)
}

/// `parse` with any token decoder.
pub fn parse_with(ids: &[u32], decode: &dyn Fn(&[u32]) -> String, ref_base: u32, n_obj: usize) -> Record {
    let mut text = String::new();
    let mut run = Vec::new();
    for &id in ids {
        if id >= ref_base {
            text.push_str(&decode(&run));
            run.clear();
            text.push_str(&format!("\u{1}{}\u{2}", id - ref_base));
        } else {
            run.push(id);
        }
    }
    text.push_str(&decode(&run));
    let mut rec = Record::default();
    if let Some(a) = text.find("<think>") {
        let body = &text[a + 7..];
        let (trace, rest) = match body.find("</think>") {
            Some(b) => (&body[..b], &body[b + 8..]),
            None => (body, ""),
        };
        rec.reasoning = trace.trim().to_string();
        text = rest.to_string();
    }
    for line in text.split('\n') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(r) = after(line, "S") {
            for (k, v) in kv(r) {
                rec.scene.insert(k, v);
            }
            continue;
        }
        if let Some(r) = line.strip_prefix('E') {
            let n: String = r.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !n.is_empty() {
                if let Some(body) = after(&r[n.len()..], "") {
                    rec.entities.push(entity(n.parse().unwrap_or(0), body));
                    continue;
                }
            }
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        if let Some(_r) = after(line, "R") {
            if words.len() >= 4 {
                let mut m = Map::new();
                m.insert("subject".into(), id_of(words[1]));
                m.insert("predicate".into(), Value::String(words[2].into()));
                m.insert("object".into(), id_of(words[3]));
                rec.relations.push(Value::Object(m));
                continue;
            }
        }
        if let Some(r) = after(line, "T") {
            let (refs, rest) = take_refs(r);
            let tail = rest.trim_end();
            let digits: String = tail.chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
            let on = (!digits.is_empty()).then(|| digits.parse().ok()).flatten();
            for k in refs {
                if k >= n_obj {
                    rec.texts.push((k - n_obj, on));
                }
            }
            continue;
        }
        let first = words[0];
        if first.len() == 1 && first.as_bytes()[0].is_ascii_uppercase() && line[1..].starts_with(char::is_whitespace) && (words.len() == 3 || words.len() == 4) {
            let mut m = Map::new();
            m.insert("actor".into(), id_of(words[1]));
            m.insert("action".into(), Value::String(words[2].into()));
            if words.len() == 4 {
                m.insert("target".into(), id_of(words[3]));
            }
            rec.events.push(Value::Object(m));
        }
    }
    rec
}
