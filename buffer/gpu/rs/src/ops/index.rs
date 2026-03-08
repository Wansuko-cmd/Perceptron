use crate::{resource::buffer::GPUBuffer, runtime::Runtime};

pub fn gather(
    x: &GPUBuffer,
    y: &GPUBuffer,
    i: usize, j: usize, k: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = runtime.kernels.index.gather(x, y, i, j, k, result);
    runtime.dispatch(task);
}
