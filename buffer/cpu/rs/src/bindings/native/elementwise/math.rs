use crate::ops::elementwise::math;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_exp_d1(x: *const f32, result: *mut f32, size: i32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    math::exp_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_ln_d1(x: *const f32, e: f32, result: *mut f32, size: i32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    math::ln_d1(x, e, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sigmoid_d1(x: *const f32, result: *mut f32, size: i32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    math::sigmoid_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_pow_d1(x: *const f32, n: i32, result: *mut f32, size: i32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    math::pow_d1(x, n, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sqrt_d1(x: *const f32, e: f32, result: *mut f32, size: i32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    math::sqrt_d1(x, e, result);
}
