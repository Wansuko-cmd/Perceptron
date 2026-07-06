use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

thread_local! {
    static CURRENT: Cell<Option<Instant>> = const { Cell::new(None) };
}

pub struct CpuProfiler {
    acc: AtomicU64,
    enabled: bool,
}

impl CpuProfiler {
    pub fn new(enabled: bool) -> Self {
        CpuProfiler { enabled, acc: AtomicU64::new(0) }
    }

    #[must_use = "bind to a variable (e.g. `let _t`), or the timer stops immediately"]
    pub fn start(&self) -> Option<OpTimer<'_>> {
        if !self.enabled || CURRENT.get().is_some() {
            return None;
        }
        CURRENT.set(Some(Instant::now()));
        Some(OpTimer { profiler: self })
    }

    pub fn take(&self) -> u64 {
        self.acc.swap(0, Ordering::Relaxed)
    }
}

pub struct OpTimer<'a> {
    profiler: &'a CpuProfiler,
}

impl Drop for OpTimer<'_> {
    fn drop(&mut self) {
        if let Some(t0) = CURRENT.take() {
            let time = t0.elapsed().as_nanos() as u64;
            self.profiler.acc.fetch_add(time, Ordering::Relaxed);
        }
    }
}
