use crate::{resource::buffer::GPUBuffer, runtime::Runtime};

pub fn mat_mul(
    x: &GPUBuffer, trans_x: bool,
    y: &GPUBuffer, trans_y: bool,
    m: usize, n: usize, k: usize, b: usize,
    result: &GPUBuffer,
    runtime: &mut Runtime,
) {
    let device = &runtime.device;
    let task = match (trans_x, trans_y) {
        (false, false) => runtime.kernels.mat_mul.mat_mul_nn(x, y, m, n, k, b, result, device),
        (false, true) => runtime.kernels.mat_mul.mat_mul_nt(x, y, m, n, k, b, result, device),
        (true, false) => runtime.kernels.mat_mul.mat_mul_tn(x, y, m, n, k, b, result, device),
        (true, true) => runtime.kernels.mat_mul.mat_mul_tt(x, y, m, n, k, b, result, device),
    };
    runtime.dispatcher.dispatch(task, device);
}
