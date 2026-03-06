use crate::{kernels::task::CopyTask, resource::buffer::GPUBuffer, runtime::Runtime};

pub fn create(size: usize, runtime: &Runtime) -> GPUBuffer {
    GPUBuffer::create(size, &runtime.device)
}

pub fn init(value: &[f32], runtime: &Runtime) -> GPUBuffer {
    GPUBuffer::init(value, &runtime.device)
}

pub fn read_all(buffer: &GPUBuffer, runtime: &mut Runtime) -> Vec<f32> {
    let map_buffer = GPUBuffer::create_map_read(buffer.count(), &runtime.device);
    runtime.dispatch(
        CopyTask {
            src: buffer,
            src_offset: 0,
            dest: &map_buffer,
            dest_offset: 0,
            size: buffer.buffer.size(),
        },
    );
    runtime.submit();

    let mut dest = vec![0.0f32; buffer.count()];
    map_buffer.copy_into(&mut dest, &runtime.device);
    return dest;
}

pub fn write(buffer: &GPUBuffer, index: usize, value: f32, runtime: &mut Runtime) {
    runtime.submit();
    buffer.write(index, value, &runtime.queue);
}

pub fn slice(buffer: &GPUBuffer, start: usize, end: usize, runtime: &mut Runtime) -> GPUBuffer {
    let size = ((end - start) * GPUBuffer::SIZE_BYTES) as u64;
    let offset = (start * GPUBuffer::SIZE_BYTES) as u64;
    let dest = create(size as usize, runtime);

    runtime.dispatch(
        CopyTask {
            src: buffer,
            src_offset: offset,
            dest: &dest,
            dest_offset: 0,
            size: size,
        },
    );

    return dest;
}

pub fn copy_into(src: &GPUBuffer, dest: &GPUBuffer, dest_offset: usize, runtime: &mut Runtime) {
    let size = src.buffer.size();
    let offset = (dest_offset * GPUBuffer::SIZE_BYTES) as u64;

    runtime.dispatch(
        CopyTask {
            src: src,
            src_offset: 0,
            dest: &dest,
            dest_offset: offset,
            size: size,
        },
    );
}

pub fn content_equals(x: &GPUBuffer, y: &GPUBuffer, runtime: &mut Runtime) -> bool {
    if x == y { return true; }
    if x.count() != y.count() { return false; }

    return read_all(x, runtime) == read_all(y, runtime);
}
