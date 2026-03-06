use crate::{resource::buffer::GPUBuffer, runtime::Runtime};

pub fn create(size: usize, runtime: &Runtime) -> GPUBuffer {
    GPUBuffer::create(size, &runtime.device)
}

pub fn init(value: &[f32], runtime: &Runtime) -> GPUBuffer {
    GPUBuffer::init(value, &runtime.device)
}

pub fn read_all(buffer: &GPUBuffer, runtime: &mut Runtime) -> Vec<f32> {
    runtime.submit();

    let mut dest = vec![0.0f32; buffer.count()];
    buffer.read_all(&mut dest, &runtime.device, &runtime.queue);

    return dest;
}

pub fn write(buffer: &GPUBuffer, index: usize, value: f32, runtime: &mut Runtime) {
    runtime.submit();
    buffer.write(index, value, &runtime.queue);
}

pub fn slice(buffer: &GPUBuffer, start: usize, end: usize, runtime: &mut Runtime) -> GPUBuffer {
    runtime.submit();
    buffer.slice(start, end, &runtime.device, &runtime.queue)
}

pub fn copy_into(src: &GPUBuffer, dest: &GPUBuffer, dest_offset: usize, runtime: &mut Runtime) {
    runtime.submit();
    src.copy_into(dest, dest_offset, &runtime.device, &runtime.queue);
}

pub fn content_equals(x: &GPUBuffer, y: &GPUBuffer, runtime: &mut Runtime) -> bool {
    runtime.submit();
    x.content_equals(y, &runtime.device, &runtime.queue)
}
