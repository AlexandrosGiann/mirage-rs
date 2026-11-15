//! Pure Rust inference for [Mirage](https://cactuscompute.com/blog/mirage), the
//! 51M-parameter on-device vision model from Cactus Compute.
//!
//! ```no_run
//! let model = mirage::Mirage::open("mirage.cact")?;
//! let img = mirage::image::decode(&std::fs::read("photo.jpg")?)?;
//! let out = model.describe(&img, &mirage::Options::default(), &mut |_| {})?;
//! println!("{}", serde_json::to_string_pretty(&out.result)?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Kernels index several parallel arrays by position; the loop form mirrors the equations.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments, clippy::unnecessary_sort_by)]

pub mod cact;
mod cactmap;
pub mod decoder;
pub mod image;
pub mod masks;
mod mirage;
pub mod ops;
pub mod par;
pub mod reader;
pub mod record;
pub mod tokenizer;
pub mod vision;

pub use crate::mirage::{Mirage, Options, Output, Proposal, Timing, FIELDS, LANGUAGES, SECTIONS};
pub use serde_json;

/// Pinned weights: Hugging Face `Cactus-Compute/mirage`, Apache-2.0.
pub const WEIGHTS_URL: &str = "https://huggingface.co/Cactus-Compute/mirage/resolve/76c1e9e53381aca8fc3843c636216c96ff220122/mirage.cact";
pub const WEIGHTS_SIZE: u64 = 21_611_043;
pub const WEIGHTS_SHA256: &str = "4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237";

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

/// Looks for `mirage.cact`: `MIRAGE_WEIGHTS`, then `./mirage.cact`, `./models/mirage.cact`,
/// and the same names in the folders above the executable.
pub fn find_weights() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("MIRAGE_WEIGHTS") {
        return Some(p.into());
    }
    let mut dirs = vec![std::path::PathBuf::from(".")];
    if let Ok(exe) = std::env::current_exe() {
        dirs.extend(exe.ancestors().skip(1).take(4).map(|p| p.to_path_buf()));
    }
    for d in dirs {
        for rel in ["mirage.cact", "models/mirage.cact"] {
            let p = d.join(rel);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}
