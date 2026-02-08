use crate::mat_mul;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_inner(x: *const f32, y: *const f32, size: usize, b: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let y = unsafe { std::slice::from_raw_parts(y, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, b) };

    mat_mul::inner(x, y, b, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_mat_mul_d1_to_d2(x: *const f32, y: *const f32, trans_y: bool, n: usize, k: usize, result: *mut f32) {
    let x_size = k;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = n * k;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result_size = n;
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };

    mat_mul::mat_mul_d1_to_d2(x, y, trans_y, n, k, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_mat_mul_d2_to_d1(x: *const f32, trans_x: bool, y: *const f32, m: usize, k: usize, result: *mut f32) {
    let x_size = m * k;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = k;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result_size = m;
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };

    mat_mul::mat_mul_d2_to_d1(x, trans_x, y, m, k, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_mat_mul_d2_to_d2(
    x: *const f32, trans_x: bool,
    y: *const f32, trans_y: bool,
    m: usize, k: usize, n: usize, b: usize,
    result: *mut f32,
) {
    let x_size = m * k * b;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = k * n * b;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result_size = m * n * b;
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };

    mat_mul::mat_mul_d2_to_d2(x, trans_x, y, trans_y, m, n, k, b, result);
}
