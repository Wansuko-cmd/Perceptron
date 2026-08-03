use crate::ops::linalg;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_inner(x: *const f32, y: *const f32, size: i32, b: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, b as usize) };
    linalg::inner(x, y, b as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_mat_mul_d1_to_d2(x: *const f32, y: *const f32, trans_y: bool, n: i32, k: i32, result: *mut f32) {
    let x_size = k;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = n * k;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result_size = n;
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    linalg::mat_mul_d1_to_d2(x, y, trans_y, n as usize, k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_mat_mul_d2_to_d1(x: *const f32, trans_x: bool, y: *const f32, m: i32, k: i32, result: *mut f32) {
    let x_size = m * k;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = k;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result_size = m;
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    linalg::mat_mul_d2_to_d1(x, trans_x, y, m as usize, k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_mat_mul_d2_to_d2(
    x: *const f32, trans_x: bool,
    y: *const f32, trans_y: bool,
    m: i32, n: i32, k: i32, b: i32,
    result: *mut f32,
) {
    let x_size = m * k * b;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = k * n * b;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result_size = m * n * b;
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    linalg::mat_mul_d2_to_d2(x, trans_x, y, trans_y, m as usize, n as usize, k as usize, b as usize, result);
}
