use crate::math;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_exp_d1(x: *const f32, result: *mut f32, size: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    math::exp_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_ln_d1(x: *const f32, e: f32, result: *mut f32, size: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    math::ln_d1(x, e, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_pow_d1(x: *const f32, n: i32, result: *mut f32, size: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    math::pow_d1(x, n, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sqrt_d1(x: *const f32, e: f32, result: *mut f32, size: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    math::sqrt_d1(x, e, result);
}
