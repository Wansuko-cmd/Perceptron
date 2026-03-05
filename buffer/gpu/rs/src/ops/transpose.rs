use crate::{resource::buffer::GPUBuffer, runtime::Runtime};

pub fn transpose_d2(x: &GPUBuffer, xi: usize, xj: usize, result: &GPUBuffer, runtime: &mut Runtime) {
    let device = &runtime.device;
    let task = runtime.kernels.transpose.transpose_d2(x, xi, xj, result, device);
    runtime.dispatch(task);
}

pub fn transpose_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis_i: usize, axis_j: usize, axis_k: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let device = &runtime.device;
    let task = runtime.kernels.transpose.transpose_d3(x, xi, xj, xk, axis_i, axis_j, axis_k, result, device);
    runtime.dispatch(task);
}

pub fn transpose_d4(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let device = &runtime.device;
    let task = runtime.kernels.transpose.transpose_d4(x, xi, xj, xk, xl, axis_i, axis_j, axis_k, axis_l, result, device);
    runtime.dispatch(task);
}
