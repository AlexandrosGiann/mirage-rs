//! `mirage` command line: describe an image, or serve the demo page and HTTP API.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mirage::serde_json::{self, json, Value};
use mirage::{image, Mirage, Options};

const USAGE: &str = "\
mirage: pure Rust inference for Cactus Mirage

usage:
  mirage <image.png|jpg> [--options JSON] [--model mirage.cact] [--stream] [--compact]
  mirage serve [--host 127.0.0.1] [--port 3000] [--model mirage.cact]

--options  describe options as JSON, e.g. '{\"sections\":[\"objects\",\"text\"]}'
--stream   print the decoder output to stderr while it is generated
--model    path to mirage.cact (default: MIRAGE_WEIGHTS, ./mirage.cact, ./models/mirage.cact)

Download the weights from
  https://huggingface.co/Cactus-Compute/mirage/resolve/main/mirage.cact
";

const INDEX: &str = include_str!("../web/index.html");
const SAMPLE: &[u8] = include_bytes!("../tests/data/sample.png");
const MAX_BODY: usize = 25 * 1024 * 1024;
const MAX_QUEUE: usize = 8;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &mut Vec<String>, name: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == name) {
        Some(i) if i + 1 < args.len() => {
            let v = args.remove(i + 1);
            args.remove(i);
            Ok(Some(v))
        }
        Some(_) => Err(format!("{name} needs a value")),
        None => Ok(None),
    }
}

fn switch(args: &mut Vec<String>, name: &str) -> bool {
    match args.iter().position(|a| a == name) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

fn load(model: Option<String>) -> Result<(Mirage, PathBuf), String> {
    let path = model.map(PathBuf::from).or_else(mirage::find_weights).ok_or_else(|| {
        format!("mirage.cact not found. Download it from {} and pass --model or put it next to the binary", mirage::WEIGHTS_URL)
    })?;
    let t = Instant::now();
    let m = Mirage::open(&path).map_err(|e| e.to_string())?;
    eprintln!("model loaded from {} in {} ms ({} threads)", path.display(), t.elapsed().as_millis(), mirage::par::threads());
    Ok((m, path))
}

fn run(mut args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || switch(&mut args, "--help") || switch(&mut args, "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let model = flag(&mut args, "--model")?;
    if args[0] == "serve" {
        args.remove(0);
        let host = flag(&mut args, "--host")?.or_else(|| std::env::var("HOST").ok()).unwrap_or_else(|| "127.0.0.1".into());
        let port = flag(&mut args, "--port")?.or_else(|| std::env::var("PORT").ok()).unwrap_or_else(|| "3000".into());
        if let Some(a) = args.first() {
            return Err(format!("unexpected argument {a}"));
        }
        let (m, path) = load(model)?;
        return serve(m, path, &host, &port);
    }
    let opts = flag(&mut args, "--options")?;
    let stream = switch(&mut args, "--stream");
    let compact = switch(&mut args, "--compact");
    if args.len() != 1 {
        return Err(format!("expected one image path\n\n{USAGE}"));
    }
    let o = match opts {
        Some(s) => Options::from_json(&serde_json::from_str(&s).map_err(|e| format!("--options is not JSON: {e}"))?).map_err(|e| e.to_string())?,
        None => Options::default(),
    };
    let bytes = std::fs::read(&args[0]).map_err(|e| format!("cannot read {}: {e}", args[0]))?;
    let img = image::decode(&bytes).map_err(|e| e.to_string())?;
    let (m, _) = load(model)?;
    let mut text = TokenText::new(&m);
    if stream {
        eprint!("<think>lang: {}", o.language);
    }
    let out = m
        .describe(&img, &o, &mut |t| {
            if stream {
                eprint!("{}", text.push(t, false));
                let _ = std::io::stderr().flush();
            }
        })
        .map_err(|e| e.to_string())?;
    if stream {
        eprintln!("{}", text.finish());
    }
    let t = out.timing;
    eprintln!("perception {} ms, decoding {} ms, assembly {} ms", t.perceive_ms, t.decode_ms, t.assemble_ms);
    let s = if compact { serde_json::to_string(&out.result) } else { serde_json::to_string_pretty(&out.result) };
    println!("{}", s.map_err(|e| e.to_string())?);
    Ok(())
}

/// Turns generated token ids into text increments (whole UTF-8 characters,
/// reference tokens shown as ⟨rN⟩).
struct TokenText<'a> {
    m: &'a Mirage,
    run: Vec<u32>,
    sent: usize,
}

impl<'a> TokenText<'a> {
    fn new(m: &'a Mirage) -> Self {
        TokenText { m, run: Vec::new(), sent: 0 }
    }

    fn flush(&mut self, fin: bool) -> String {
        let text = self.m.tokenizer.decode(&self.run);
        if !fin && text.ends_with('\u{fffd}') {
            return String::new();
        }
        let out = text.get(self.sent..).unwrap_or("").to_string();
        self.sent = text.len().max(self.sent);
        out
    }

    fn push(&mut self, t: u32, _fin: bool) -> String {
        if t >= self.m.ref_base {
            let mut s = self.flush(true);
            s.push_str(&format!("\u{27e8}r{}\u{27e9}", t - self.m.ref_base));
            self.run.clear();
            self.sent = 0;
            return s;
        }
        self.run.push(t);
        self.flush(false)
    }

    fn finish(&mut self) -> String {
        self.flush(true)
    }
}

// ---- HTTP -------------------------------------------------------------------

struct Server {
    model: Mirage,
    weights: String,
    gate: Mutex<()>,
    waiting: AtomicUsize,
}

fn serve(model: Mirage, path: PathBuf, host: &str, port: &str) -> Result<(), String> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
    let shown = if host == "0.0.0.0" { format!("localhost:{port}") } else { addr.clone() };
    eprintln!("mirage-rs {} listening on http://{shown}", env!("CARGO_PKG_VERSION"));
    let srv = Arc::new(Server {
        model,
        weights: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        gate: Mutex::new(()),
        waiting: AtomicUsize::new(0),
    });
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let srv = srv.clone();
        std::thread::spawn(move || {
            let _ = handle(&srv, stream);
        });
    }
    Ok(())
}

struct Request {
    method: String,
    path: String,
    query: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
    fn param(&self, k: &str) -> Option<String> {
        self.query.split('&').find_map(|kv| {
            let (a, b) = kv.split_once('=').unwrap_or((kv, ""));
            (pct_decode(a) == k).then(|| pct_decode(b))
        })
    }
}

fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_request(stream: &mut TcpStream) -> Result<Request, (u16, String)> {
    let mut r = BufReader::new(stream.try_clone().map_err(|e| (500, e.to_string()))?);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| (400, e.to_string()))?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let (path, query) = target.split_once('?').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((target.clone(), String::new()));
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).map_err(|e| (400, e.to_string()))? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
        if headers.len() > 100 {
            return Err((431, "too many headers".into()));
        }
    }
    let mut req = Request { method, path, query, headers, body: Vec::new() };
    if let Some(n) = req.header("content-length") {
        let n: usize = n.parse().map_err(|_| (400, "bad content-length".to_string()))?;
        if n > MAX_BODY {
            return Err((413, "image too large (max 25 MB)".into()));
        }
        req.body = vec![0; n];
        r.read_exact(&mut req.body).map_err(|e| (400, e.to_string()))?;
    } else if req.header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        loop {
            let mut sz = String::new();
            r.read_line(&mut sz).map_err(|e| (400, e.to_string()))?;
            let n = usize::from_str_radix(sz.trim().split(';').next().unwrap_or(""), 16).map_err(|_| (400, "bad chunk".to_string()))?;
            if n == 0 {
                break;
            }
            if req.body.len() + n > MAX_BODY {
                return Err((413, "image too large (max 25 MB)".into()));
            }
            let s = req.body.len();
            req.body.resize(s + n, 0);
            r.read_exact(&mut req.body[s..]).map_err(|e| (400, e.to_string()))?;
            let mut crlf = [0u8; 2];
            r.read_exact(&mut crlf).map_err(|e| (400, e.to_string()))?;
        }
    }
    Ok(req)
}

fn reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8], head_only: bool) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 {code} {}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n", reason(code), body.len())?;
    if !head_only {
        s.write_all(body)?;
    }
    s.flush()
}

fn send_json(s: &mut TcpStream, code: u16, v: &Value) -> std::io::Result<()> {
    respond(s, code, "application/json; charset=utf-8", v.to_string().as_bytes(), false)
}

fn handle(srv: &Server, mut s: TcpStream) -> std::io::Result<()> {
    let req = match read_request(&mut s) {
        Ok(r) => r,
        Err((code, msg)) => return send_json(&mut s, code, &json!({ "error": msg })),
    };
    let head = req.method == "HEAD";
    match (req.method.as_str(), req.path.as_str()) {
        ("GET" | "HEAD", "/" | "/index.html") => respond(&mut s, 200, "text/html; charset=utf-8", INDEX.as_bytes(), head),
        ("GET" | "HEAD", "/sample.png") => respond(&mut s, 200, "image/png", SAMPLE, head),
        ("GET", "/api/health") => send_json(
            &mut s,
            200,
            &json!({ "name": "mirage-rs", "version": env!("CARGO_PKG_VERSION"), "ready": true, "error": null, "queue": srv.waiting.load(Ordering::SeqCst), "weights": srv.weights }),
        ),
        ("POST", "/api/describe") => describe(srv, &mut s, &req),
        ("GET" | "HEAD", _) => send_json(&mut s, 404, &json!({ "error": "not found" })),
        _ => send_json(&mut s, 405, &json!({ "error": "method not allowed" })),
    }
}

fn describe(srv: &Server, s: &mut TcpStream, req: &Request) -> std::io::Result<()> {
    let raw = req.param("options").or_else(|| req.header("x-mirage-options").map(String::from)).unwrap_or_else(|| "{}".into());
    let o = match serde_json::from_str::<Value>(&raw).map_err(|e| e.to_string()).and_then(|v| Options::from_json(&v).map_err(|e| e.to_string())) {
        Ok(o) => o,
        Err(e) => return send_json(s, 400, &json!({ "error": format!("invalid options: {e}") })),
    };
    if req.body.is_empty() {
        return send_json(s, 400, &json!({ "error": "empty body: POST the PNG or JPEG bytes" }));
    }
    let img = match image::decode(&req.body) {
        Ok(i) => i,
        Err(e) => return send_json(s, 422, &json!({ "error": e.to_string() })),
    };
    let position = srv.waiting.fetch_add(1, Ordering::SeqCst);
    if position >= MAX_QUEUE {
        srv.waiting.fetch_sub(1, Ordering::SeqCst);
        return send_json(s, 503, &json!({ "error": "server busy, try again shortly" }));
    }
    struct Leave<'a>(&'a AtomicUsize);
    impl Drop for Leave<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _leave = Leave(&srv.waiting);
    let stream = req.header("accept").is_some_and(|a| a.contains("application/x-ndjson"));
    let started = Instant::now();
    if stream {
        write!(s, "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson; charset=utf-8\r\ncache-control: no-store\r\nx-accel-buffering: no\r\nconnection: close\r\n\r\n")?;
        writeln!(s, "{}", json!({ "type": "queued", "position": position }))?;
        s.flush()?;
    }
    let _turn = srv.gate.lock().unwrap_or_else(|p| p.into_inner());
    let mut out = s.try_clone()?;
    let emit = |out: &mut TcpStream, v: Value| {
        if stream {
            let _ = writeln!(out, "{v}");
            let _ = out.flush();
        }
    };
    emit(&mut out, json!({ "type": "stage", "stage": "perceive" }));
    let mut text = TokenText::new(&srv.model);
    let mut started_decode = false;
    let lang = o.language.clone();
    let res = srv.model.describe(&img, &o, &mut |t| {
        if !started_decode {
            started_decode = true;
            emit(&mut out, json!({ "type": "stage", "stage": "decode" }));
            emit(&mut out, json!({ "type": "token", "text": format!("<think>lang: {lang}") }));
        }
        let piece = text.push(t, false);
        if !piece.is_empty() {
            emit(&mut out, json!({ "type": "token", "text": piece }));
        }
    });
    let tail = text.finish();
    if !tail.is_empty() {
        emit(&mut out, json!({ "type": "token", "text": tail }));
    }
    match res {
        Ok(r) => {
            let t = r.timing;
            if stream {
                let meta = json!({ "timing": { "perceive_ms": t.perceive_ms, "decode_ms": t.decode_ms, "assemble_ms": t.assemble_ms }, "proposals": { "objects": r.proposals.0, "text": r.proposals.1 }, "total_ms": started.elapsed().as_millis() as u64 });
                emit(&mut out, json!({ "type": "result", "result": r.result, "meta": meta }));
                Ok(())
            } else {
                send_json(s, 200, &r.result)
            }
        }
        Err(e) => {
            if stream {
                emit(&mut out, json!({ "type": "error", "error": e.to_string() }));
                Ok(())
            } else {
                send_json(s, 422, &json!({ "error": e.to_string() }))
            }
        }
    }
}
