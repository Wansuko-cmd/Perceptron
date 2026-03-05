use crate::runtime::Runtime;

pub fn runtime_allocate() -> Runtime {
    pollster::block_on(Runtime::new())
}
