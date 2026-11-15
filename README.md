# mirage-rs

Pure Rust inference for **[Mirage](https://cactuscompute.com/blog/mirage)**, the 51M-parameter on-device vision model from Cactus Compute. One call turns an image into a structured record: a scene description, named objects with boxes, instance masks, OCR text, relations and a short reasoning trace.

No C, no WebAssembly, no Python. The model reads the official 2-bit `mirage.cact` weights (21.6 MB, Apache-2.0) directly. The only dependencies are pure-Rust crates: `image` (PNG and JPEG decoding through `png` and `zune-jpeg`) and `serde_json`. The HTTP server uses only `std`.

> Unofficial port. It is a translation of the pure JavaScript port [mirage-js](https://github.com/AlexandrosGiann/mirage-js) and produces the same output (see [Verification](#verification)). The reference engine (`libcactus`) is distributed as a binary, so both ports were rebuilt from the published forward-pass equations, the released weights and the open-source Cactus decoder code.

## Get the weights

The model is not in this repository. Download `mirage.cact` from Hugging Face:

<https://huggingface.co/Cactus-Compute/mirage/resolve/main/mirage.cact>

Put it in this folder (or `models/`), or pass `--model`, or set `MIRAGE_WEIGHTS`. The tested revision is `76c1e9e5`, SHA-256 `4249a0f7…f237`.

## Build and run

Needs a Rust toolchain (<https://rustup.rs>; written against Rust 1.99, builds on 1.80 or newer).

```bash
cargo build --release
```

```bash
target/release/mirage photo.jpg
```

```bash
target/release/mirage photo.jpg --options '{"sections":["objects","text"],"mask_format":"polygons"}' --stream
```

`--stream` prints the decoder output to stderr while it is generated. The JSON record goes to stdout (`--compact` for one line).

### Demo page and HTTP API

```bash
target/release/mirage serve --port 3000            # http://127.0.0.1:3000
target/release/mirage serve --host 0.0.0.0         # reachable from other devices on your network
```

The page takes an image, streams the decoder output, and draws the boxes and masks. The API is the same as mirage-js:

- `POST /api/describe` with the PNG or JPEG bytes as the body. Options go in the `options` query parameter (URL-encoded JSON).
- `Accept: application/json` returns the record.
- `Accept: application/x-ndjson` streams `queued`, `stage`, `token` and `result` events.
- `GET /api/health` reports the server status.

Requests are served one at a time, with up to 8 waiting.

## Use it from code

```rust
let model = mirage::Mirage::open("mirage.cact")?;
let img = mirage::image::decode(&std::fs::read("photo.jpg")?)?;   // or mirage::image::Rgb::new(w, h, rgb)
let opts = mirage::Options::from_json(&serde_json::json!({ "mask_format": "polygons" }))?;
let out = model.describe(&img, &opts, &mut |token_id| { /* streamed tokens */ })?;
println!("{}", serde_json::to_string_pretty(&out.result)?);
```

- `describe` returns `Output { result, raw, timing, proposals }`. `result` is the JSON record, and `raw` is the decoder text.
- `model.detect(&img, &opts)` returns the detector proposals only, without running the decoder.
- `Mirage` is `Send + Sync` and `describe` takes `&self`. The decoder state is per call, so one loaded model can serve several threads.
- Kernels use all cores. Set `MIRAGE_THREADS` to limit them.

## Options

| option | default | meaning |
|---|---|---|
| `sections` | all | `scene`, `objects`, `regions`, `text`, `relations`, `events`, `reasoning` |
| `fields` | `bbox score mask position description` | also `point`, `count` |
| `mask_format` | `largest` | `largest` (one polygon), `polygons`, `merged`, `bitmap` (COCO RLE, column-major) |
| `conf` / `text_conf` | 0.3 | minimum detector score for object / text proposals |
| `max_objects` / `max_text_lines` | 64 / 32 | proposal budgets |
| `max_tokens` | 768 | decoder budget |
| `language` | `en` | `en de fr es it nl pl` |
| `mask_threshold` / `mask_tolerance` | 0.5 / 1.5 | mask probability cut-off / polygon simplification (pixels) |
| `nms`, `iou`, `agnostic_nms` | off, 0.7, false | optional per-name box suppression |

Coordinates are integers on a 0–1000 scale of each original image axis. The output format is described in the [mirage-js README](https://github.com/AlexandrosGiann/mirage-js#output).

## How it works

The pipeline follows Mirage's published forward pass.

1. **Letterbox**: the long side becomes 512 px, and each side is rounded up to a multiple of 32.
2. **Encoder**: a convolutional stem, 16 HConvGLU body blocks and 12 attention blocks with 2D RoPE and 4 registers.
3. **Neck and heads**: a gated level memory (16×16×512) and a feature-pyramid neck (64×64×128) feed CenterNet-style proposals and YOLACT-style masks.
4. **Decoder**: box-reference slots join the image memory. The decoder has 8 layers on 4 mHC lanes, with engram memory, tapped self-attention, cross-attention and a Hadamard MLP. It writes the compact record that the runtime resolves into boxes and masks.
5. **OCR**: a CTC reader reads each text box in 48×320 windows.

The source files map one-to-one to the mirage-js modules: `cact`, `vision`, `decoder`, `reader`, `tokenizer`, `record`, `masks`, `image` and `mirage`.

## Verification

- **mirage-js parity**: on PNG inputs, the JSON record is identical to mirage-js. That includes the decoder text, boxes, scores, polygon and RLE masks, and OCR strings, checked on the sample and on two photos.
- **Official engine**: the object proposals of the official `libcactus` 4.1.1 output are found among the Rust proposals with scores within ±0.007 (4/4, 13/13 and 5/6 on three photos). The JS port gives the same match.
- **JPEG**: `zune-jpeg` decodes within 0.01/255 (mean) of PIL, which the official Python path uses. jpeg-js, which mirage-js uses, is within 0.5/255. On JPEG photos the two ports can therefore start to differ after a few tokens.
- `cargo test` runs the unit tests and, when the weights are found, an end-to-end test against the official reference values and the mirage-js output.

Speed on an 8-thread desktop CPU: about 0.3 s for perception and 2–7 s for the description. mirage-js takes about 3 s and 5–10 s.

## Limitations

- **Forms** (`options.form`) are not supported, because the engine's form prompt is not published.
- The small 2D positional term added to the image memory is unspecified (its gate is about 0.012), so it is omitted.
- Polygon tracing is this port's own. Masks agree with the official engine, but vertex lists differ.
- The library decodes PNG and JPEG. For other formats, pass `Rgb` pixels.

## Related

- [mirage-js](https://github.com/AlexandrosGiann/mirage-js): the JavaScript port this one follows. It also runs in the browser.
- [Cactus_Whistle-Pure_Rust](https://github.com/AlexandrosGiann/Cactus_Whistle-Pure_Rust) (speech) and [MobileCLIP-S0_PureRust](https://github.com/AlexandrosGiann/MobileCLIP-S0_PureRust): other pure Rust ports by the same author.

## Credits and license

Mirage, its architecture and its weights are by [Cactus Compute](https://cactuscompute.com) and are released under Apache-2.0 ([Hugging Face](https://huggingface.co/Cactus-Compute/mirage), [GitHub](https://github.com/cactus-compute/cactus)). This project is an independent reimplementation and is not affiliated with or endorsed by Cactus Compute. The code in this repository is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
