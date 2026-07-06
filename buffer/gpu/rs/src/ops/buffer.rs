
use crate::{kernels::task::CopyTask, resource::buffer::GPUBuffer, runtime::Runtime};

pub fn create(size: usize, runtime: &Runtime) -> GPUBuffer {
    let _t = runtime.cpu_profiler.start();
    GPUBuffer::create(size, &runtime.device)
}

pub fn init(value: &[f32], runtime: &Runtime) -> GPUBuffer {
    let _t = runtime.cpu_profiler.start();
    GPUBuffer::init(value, &runtime.device)
}

pub fn read_all(buffer: &GPUBuffer, runtime: &mut Runtime) -> Vec<f32> {
    let size = buffer.count();
    let map_buffer = GPUBuffer::create_map_read(size, &runtime.device);
    runtime.dispatch(
        CopyTask {
            src: buffer,
            src_offset: 0,
            dest: &map_buffer,
            dest_offset: 0,
            size: size,
        },
    );
    runtime.flush();

    let mut dest = vec![0.0f32; buffer.count()];
    map_buffer.copy_into(&mut dest, &runtime.device);
    return dest;
}

pub fn write(buffer: &GPUBuffer, index: usize, value: f32, runtime: &mut Runtime) {
    runtime.flush();
    buffer.write(index, value, &runtime.queue);
}
