use crate::{resource::buffer::GPUBuffer, runtime::Runtime};

pub fn average_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = runtime.kernels.collection.average_d3(x, 1, x.count(), 1, result);
    runtime.dispatch(task);
}

pub fn average_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.average_d3(x, 1, xi, xj, result),
        _ => runtime.kernels.collection.average_d3(x, xi, xj, 1, result),
    };
    runtime.dispatch(task);
}

pub fn average_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.average_d3(x, 1, xi, xj * xk, result),
        1 => runtime.kernels.collection.average_d3(x, xi, xj, xk, result),
        _ => runtime.kernels.collection.average_d3(x, xi * xj, xk, 1, result),
    };
    runtime.dispatch(task);
}

pub fn max_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = runtime.kernels.collection.max_d3(x, 1, x.count(), 1, result);
    runtime.dispatch(task);
}

pub fn max_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.max_d3(x, 1, xi, xj, result),
        _ => runtime.kernels.collection.max_d3(x, xi, xj, 1, result),
    };
    runtime.dispatch(task);
}

pub fn max_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.max_d3(x, 1, xi, xj * xk, result),
        1 => runtime.kernels.collection.max_d3(x, xi, xj, xk, result),
        _ => runtime.kernels.collection.max_d3(x, xi * xj, xk, 1, result),
    };
    runtime.dispatch(task);
}

pub fn min_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = runtime.kernels.collection.min_d3(x, 1, x.count(), 1, result);
    runtime.dispatch(task);
}

pub fn min_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.min_d3(x, 1, xi, xj, result),
        _ => runtime.kernels.collection.min_d3(x, xi, xj, 1, result),
    };
    runtime.dispatch(task);
}

pub fn min_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.min_d3(x, 1, xi, xj * xk, result),
        1 => runtime.kernels.collection.min_d3(x, xi, xj, xk, result),
        _ => runtime.kernels.collection.min_d3(x, xi * xj, xk, 1, result),
    };
    runtime.dispatch(task);
}

pub fn sum_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = runtime.kernels.collection.sum_d3(x, 1, x.count(), 1, result);
    runtime.dispatch(task);
}

pub fn sum_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.sum_d3(x, 1, xi, xj, result),
        _ => runtime.kernels.collection.sum_d3(x, xi, xj, 1, result),
    };
    runtime.dispatch(task);
}

pub fn sum_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let task = match axis {
        0 => runtime.kernels.collection.sum_d3(x, 1, xi, xj * xk, result),
        1 => runtime.kernels.collection.sum_d3(x, xi, xj, xk, result),
        _ => runtime.kernels.collection.sum_d3(x, xi * xj, xk, 1, result),
    };
    runtime.dispatch(task);
}
