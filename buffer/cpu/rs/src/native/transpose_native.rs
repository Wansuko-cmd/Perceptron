use crate::transpose;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_transpose_d2(x: *const f32, xi: usize, xj: usize, result: *mut f32) {
    let size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    transpose::transpose_d2(x, xi, xj, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_transpose_d3(
    x: *const f32,
    xi: usize, xj: usize, xk: usize,
    axis_i: usize, axis_j: usize, axis_k: usize,
    result: *mut f32,
) {
    let size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    transpose::transpose_d3(x, xi, xj, xk, axis_i, axis_j, axis_k, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_transpose_d4(
    x: *const f32,
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
    result: *mut f32,
) {
    let size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    transpose::transpose_d4(x, xi, xj, xk, xl, axis_i, axis_j, axis_k, axis_l, result);
}
