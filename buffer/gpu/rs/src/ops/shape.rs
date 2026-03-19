use crate::{ops, resource::buffer::GPUBuffer, runtime::Runtime};

pub fn transpose_d2(x: &GPUBuffer, xi: usize, xj: usize, result: &GPUBuffer, runtime: &mut Runtime) {
    let task = runtime.kernels.shape.transpose_d2(x, xi, xj, result);
    runtime.dispatch(task);
}

pub fn transpose_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis_i: usize, axis_j: usize, axis_k: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match(axis_i, axis_j, axis_k) {
        (0, 1, 2) => {
            ops::buffer::copy_into(x, &result, 0, x.count(), 1, runtime);
            return;
        },
        (1, 2, 0) => runtime.kernels.shape.transpose_d2(x, xi, xj * xk, result),
        (2, 0, 1) => runtime.kernels.shape.transpose_d2(x, xi * xj, xk, result),
        _ => runtime.kernels.shape.transpose_d3(x, xi, xj, xk, axis_i, axis_j, axis_k, result)
    };
    runtime.dispatch(task);
}

pub fn transpose_d4(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match(axis_i, axis_j, axis_k, axis_l) {
        (0, 1, 2, 3) => {
            ops::buffer::copy_into(x, &result, 0, x.count(), 1, runtime);
            return;
        },
        (1, 2, 3, 0) => runtime.kernels.shape.transpose_d2(x, xi, xj * xk * xl, result),
        (2, 3, 0, 1) => runtime.kernels.shape.transpose_d2(x, xi * xj, xk * xl, result),
        (3, 0, 1, 2) => runtime.kernels.shape.transpose_d2(x, xi * xj * xk, xl, result),
        _ => runtime.kernels.shape.transpose_d4(x, xi, xj, xk, xl, axis_i, axis_j, axis_k, axis_l, result),
    };
    runtime.dispatch(task);
}

pub fn slice_d1(x: &GPUBuffer, start: usize, end: usize, step: isize, result: &GPUBuffer, runtime: &mut Runtime) {
    let task = runtime.kernels.shape.slice_d1(x, start, end, step, result);
    runtime.dispatch(task);
}

pub fn slice_d2(
    x: &GPUBuffer, xi: usize, xj: usize, axis: usize,
    start: usize, end: usize, step: isize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.shape.slice_d2_axis0(x, xi, xj, start, end, step, result),
        _ => runtime.kernels.shape.slice_d2_axis1(x, xi, xj, start, end, step, result),
    };
    runtime.dispatch(task);
}

pub fn slice_d3(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, axis: usize,
    start: usize, end: usize, step: isize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.shape.slice_d2_axis0(x, xi, xj * xk, start, end, step, result),
        1 => runtime.kernels.shape.slice_d3(x, xi, xj, xk, start, end, step, result),
        _ => runtime.kernels.shape.slice_d2_axis1(x, xi * xj, xk, start, end, step, result),
    };
    runtime.dispatch(task);
}
