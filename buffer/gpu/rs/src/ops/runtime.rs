use crate::runtime::Runtime;

pub fn allocate(enable_profiler: bool) -> Runtime {
    pollster::block_on(Runtime::new(enable_profiler))
}
