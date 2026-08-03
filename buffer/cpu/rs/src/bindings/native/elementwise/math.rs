use crate::ops::elementwise::math;
use crate::resource::buffer::CPUBuffer;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_exp_d1(x: *const CPUBuffer, result: *mut CPUBuffer) {
    let x = unsafe { &*x };
    let result = unsafe { &mut *result };
    math::exp_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_ln_d1(x: *const CPUBuffer, e: f32, result: *mut CPUBuffer) {
    let x = unsafe { &*x };
    let result = unsafe { &mut *result };
    math::ln_d1(x, e, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sigmoid_d1(x: *const CPUBuffer, result: *mut CPUBuffer) {
    let x = unsafe { &*x };
    let result = unsafe { &mut *result };
    math::sigmoid_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_pow_d1(x: *const CPUBuffer, n: i32, result: *mut CPUBuffer) {
    let x = unsafe { &*x };
    let result = unsafe { &mut *result };
    math::pow_d1(x, n, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sqrt_d1(x: *const CPUBuffer, e: f32, result: *mut CPUBuffer) {
    let x = unsafe { &*x };
    let result = unsafe { &mut *result };
    math::sqrt_d1(x, e, result);
}
