use crate::runtime::Runtime;
use crate::resource::buffer::CPUBuffer;

pub fn create(size: usize, runtime: &Runtime) -> CPUBuffer {
    runtime.create_buffer(size)
}

pub fn init(value: &[f32], runtime: &Runtime) -> CPUBuffer {
    runtime.init_buffer(value)
}

pub fn release(buffer: CPUBuffer, runtime: &Runtime) {
    runtime.release_buffer(buffer);
}
