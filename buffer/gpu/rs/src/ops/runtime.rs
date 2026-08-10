use crate::runtime::Runtime;

pub fn allocate(pool_size: u64, enable_profiler: bool) -> Runtime {
    pollster::block_on(Runtime::new(pool_size, enable_profiler))
}
