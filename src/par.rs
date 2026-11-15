//! Fork-join helpers on scoped std threads.
//!
//! Kernels split their output into contiguous row blocks and run the blocks on
//! `std::thread::scope` threads. Small calls stay on the calling thread, because
//! starting threads costs more than the work itself.

use std::sync::OnceLock;

/// Smallest kernel call, in multiply-adds, worth splitting across threads.
pub const MIN_WORK: usize = 1 << 21;

static THREADS: OnceLock<usize> = OnceLock::new();

/// Number of worker threads: `MIRAGE_THREADS`, else every available core.
pub fn threads() -> usize {
    *THREADS.get_or_init(|| {
        std::env::var("MIRAGE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
    })
}

/// Splits `out` into rows of `row` values and calls `f(first_row, block)` on
/// contiguous blocks, in parallel when `work` (multiply-adds) is large enough.
pub fn rows<F>(out: &mut [f32], row: usize, work: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    rows_of(out, row, work, f)
}

/// `rows` for any element type.
pub fn rows_of<T: Send, F>(out: &mut [T], row: usize, work: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    let row = row.max(1);
    let n = out.len() / row;
    let t = if work < MIN_WORK { 1 } else { threads().min(n) };
    if t <= 1 || n <= 1 {
        f(0, out);
        return;
    }
    let per = n.div_ceil(t);
    std::thread::scope(|s| {
        let f = &f;
        let mut rest = out;
        let mut r0 = 0;
        while !rest.is_empty() {
            let take = (per * row).min(rest.len());
            let (a, b) = rest.split_at_mut(take);
            rest = b;
            let start = r0;
            r0 += take / row;
            if rest.is_empty() {
                f(start, a);
            } else {
                s.spawn(move || f(start, a));
            }
        }
    });
}
