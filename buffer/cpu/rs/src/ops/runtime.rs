use crate::runtime::Runtime;

pub fn allocate(pool_size: u64) -> Runtime {
    pollster::block_on(Runtime::new(pool_size))
}
