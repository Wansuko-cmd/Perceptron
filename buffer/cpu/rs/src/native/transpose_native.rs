use crate::transpose;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_transpose_d2(x: *const f32, xi: i32, xj: i32, result: *mut f32) {
    let size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    transpose::transpose_d2(x, xi as usize, xj as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_transpose_d3(
    x: *const f32,
    xi: i32, xj: i32, xk: i32,
    axis_i: i32, axis_j: i32, axis_k: i32,
    result: *mut f32,
) {
    let size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    transpose::transpose_d3(x, xi as usize, xj as usize, xk as usize, axis_i as usize, axis_j as usize, axis_k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_transpose_d4(
    x: *const f32,
    xi: i32, xj: i32, xk: i32, xl: i32,
    axis_i: i32, axis_j: i32, axis_k: i32, axis_l: i32,
    result: *mut f32,
) {
    let size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    transpose::transpose_d4(x, xi as usize, xj as usize, xk as usize, xl as usize, axis_i as usize, axis_j as usize, axis_k as usize, axis_l as usize, result);
}
