pub mod r#where;

use crate::ops::elementwise::compare;

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_greater_than_d1_to_d0(x: *const f32, size: i32, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    compare::greater_than_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_greater_than_d1_to_d1(x: *const f32, y: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    compare::greater_than_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_less_than_d1_to_d0(x: *const f32, size: i32, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    compare::less_than_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_less_than_d1_to_d1(x: *const f32, y: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    compare::less_than_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_equals_d1_to_d0(x: *const f32, size: i32, y: f32, atol: f32, rtol: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    compare::equals_d1_to_d0(x, y, atol, rtol, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_equals_d1_to_d1(x: *const f32, y: *const f32, size: i32, atol: f32, rtol: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    compare::equals_d1_to_d1(x, y, atol, rtol, result);
}
