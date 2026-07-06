use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
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

pub struct GpuProfiler {
    enabled: bool,
    query_set: Option<wgpu::QuerySet>,
    query_set_index: AtomicU32,
}

impl GpuProfiler {
    const TIMESTAMP_CAPACITY: u32 = 4096;

    pub fn new(device: &wgpu::Device, enabled: bool) -> Self {
        if !enabled {
            return GpuProfiler { enabled, query_set: None, query_set_index: AtomicU32::new(0) };
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Profiler::new"),
            ty: wgpu::QueryType::Timestamp,
            count: Self::TIMESTAMP_CAPACITY,
        });
        GpuProfiler {
            enabled: enabled,
            query_set: Some(query_set),
            query_set_index: AtomicU32::new(0),
        }
    }

    pub fn start(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.enabled {
            return None;
        }
        let begin = self.query_set_index.fetch_add(2, Ordering::Relaxed);
        let end = begin + 1;
        if end >= Self::TIMESTAMP_CAPACITY {
            return None;
        }

        let query_set = self.query_set.as_ref().unwrap();
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }
}
