use crate::{resource::buffer::GPUBuffer, runtime::Runtime};

pub fn exp_d1(x: &GPUBuffer, result: &GPUBuffer, runtime: &mut Runtime) {
    let device = &runtime.device;
    let task = runtime.kernels.math.exp_d1(x, result, device);
    runtime.dispatcher.dispatch(task, device);
}

pub fn ln_d1(x: &GPUBuffer, e: f32, result: &GPUBuffer, runtime: &mut Runtime) {
    let device = &runtime.device;
    let task = runtime.kernels.math.ln_d1(x, e, result, device);
    runtime.dispatcher.dispatch(task, device);
}

pub fn sigmoid_d1(x: &GPUBuffer, result: &GPUBuffer, runtime: &mut Runtime) {
    let device = &runtime.device;
    let task = runtime.kernels.math.sigmoid_d1(x, result, device);
    runtime.dispatcher.dispatch(task, device);
}

pub fn pow_d1(x: &GPUBuffer, n: i32, result: &GPUBuffer, runtime: &mut Runtime) {
    let device = &runtime.device;
    let task = runtime.kernels.math.pow_d1(x, n as f32, result, device);
    runtime.dispatcher.dispatch(task, device);
}

pub fn sqrt_d1(x: &GPUBuffer, e: f32, result: &GPUBuffer, runtime: &mut Runtime) {
    let device = &runtime.device;
    let task = runtime.kernels.math.sqrt_d1(x, e, result, device);
    runtime.dispatcher.dispatch(task, device);
}
