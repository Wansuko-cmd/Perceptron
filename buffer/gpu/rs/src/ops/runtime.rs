use crate::runtime::Runtime;

pub fn allocate() -> Runtime {
    pollster::block_on(Runtime::new())
}
