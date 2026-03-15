use std::cmp::min;

use crate::{kernels::task::CopyTask, resource::buffer::GPUBuffer, runtime::Runtime};

pub fn create(size: usize, runtime: &Runtime) -> GPUBuffer {
    GPUBuffer::create(size, &runtime.device)
}

pub fn init(value: &[f32], runtime: &Runtime) -> GPUBuffer {
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
    runtime.submit();

    let mut dest = vec![0.0f32; buffer.count()];
    map_buffer.copy_into(&mut dest, &runtime.device);
    return dest;
}

pub fn write(buffer: &GPUBuffer, index: usize, value: f32, runtime: &mut Runtime) {
    runtime.submit();
    buffer.write(index, value, &runtime.queue);
}

pub fn slice(buffer: &GPUBuffer, start: usize, end: usize, step: isize, runtime: &mut Runtime) -> GPUBuffer {
    let size = ((end as isize - start as isize) / step  + 1) as usize;
    let dest = create(size, runtime);
    if step == 1 {
        runtime.dispatch(
            CopyTask {
                src: buffer,
                src_offset: start,
                dest: &dest,
                dest_offset: 0,
                size: size,
            },
        );
    } else {
        runtime.dispatch(runtime.kernels.shape.slice(buffer, start, step, &dest));
    }

    return dest;
}

pub fn copy_into(src: &GPUBuffer, dest: &GPUBuffer, start: usize, end: usize, step: isize, runtime: &mut Runtime) {
    let dest_size = (end as isize - start as isize) as isize / step + 1;
    let size = min(src.count(), dest_size as usize);
    match step {
        0 => {
            runtime.dispatch(
                CopyTask {
                    src: src,
                    src_offset: 0,
                    dest: &dest,
                    dest_offset: start,
                    size: size,
                },
            );
        }
        _ => {
            runtime.dispatch(runtime.kernels.shape.copy_into(src, start, step, size, dest));
        }
    }
}

pub fn content_equals(x: &GPUBuffer, y: &GPUBuffer, runtime: &mut Runtime) -> bool {
    if x == y { return true; }
    if x.count() != y.count() { return false; }

    return read_all(x, runtime) == read_all(y, runtime);
}
