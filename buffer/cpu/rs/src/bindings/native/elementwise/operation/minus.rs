use crate::ops::elementwise::operation::minus;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d0_to_d1(x: f32, y: *const f32, y_size: i32, result: *mut f32) {
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size as usize) };
    minus::minus_with_d0_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d0(x: *const f32, x_size: i32, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d1(x: *const f32, y: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    minus::minus_with_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d2(
    x: *const f32,
    y: *const f32, yi: i32, yj: i32,
    axis: i32,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi as usize,
        1 => yj as usize,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };
    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size as usize) };
    minus::minus_with_d1_to_d2(x, y, yi as usize, yj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d3(
    x: *const f32,
    y: *const f32, yi: i32, yj: i32, yk: i32,
    axis: i32,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi as usize,
        1 => yj as usize,
        2 => yk as usize,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };
    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size as usize) };
    minus::minus_with_d1_to_d3(x, y, yi as usize, yj as usize, yk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d2_to_d1(
    x: *const f32, xi: i32, xj: i32,
    y: *const f32,
    axis: i32,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = match axis {
        0 => xi as usize,
        1 => xj as usize,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d2_to_d1(x, xi as usize, xj as usize, y, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d2_to_d3(
    x: *const f32, xi: i32, xj: i32,
    y: *const f32, yi: i32, yj: i32, yk: i32,
    axis1: i32, axis2: i32,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size as usize) };
    minus::minus_with_d2_to_d3(x, xi as usize, xj as usize, y, yi as usize, yj as usize, yk as usize, axis1 as usize, axis2 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d3_to_d1(
    x: *const f32, xi: i32, xj: i32, xk: i32,
    y: *const f32,
    axis: i32,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = match axis {
        0 => xi as usize,
        1 => xj as usize,
        2 => xk as usize,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d3_to_d1(x, xi as usize, xj as usize, xk as usize, y, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d3_to_d2(
    x: *const f32, xi: i32, xj: i32, xk: i32,
    y: *const f32, yi: i32, yj: i32,
    axis1: i32, axis2: i32,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d3_to_d2(x, xi as usize, xj as usize, xk as usize, y, yi as usize, yj as usize, axis1 as usize, axis2 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d3_to_d4(
    x: *const f32, xi: i32, xj: i32, xk: i32,
    y: *const f32, yi: i32, yj: i32, yk: i32, yl: i32,
    axis1: i32, axis2: i32, axis3: i32,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = yi * yj * yk * yl;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size as usize) };
    minus::minus_with_d3_to_d4(x, xi as usize, xj as usize, xk as usize, y, yi as usize, yj as usize, yk as usize, yl as usize, axis1 as usize, axis2 as usize, axis3 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d4_to_d1(
    x: *const f32, xi: i32, xj: i32, xk: i32, xl: i32,
    y: *const f32,
    axis: i32,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = match axis {
        0 => xi as usize,
        1 => xj as usize,
        2 => xk as usize,
        3 => xl as usize,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d4_to_d1(x, xi as usize, xj as usize, xk as usize, xl as usize, y, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d4_to_d2(
    x: *const f32, xi: i32, xj: i32, xk: i32, xl: i32,
    y: *const f32, yi: i32, yj: i32,
    axis1: i32, axis2: i32,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d4_to_d2(x, xi as usize, xj as usize, xk as usize, xl as usize, y, yi as usize, yj as usize, axis1 as usize, axis2 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d4_to_d3(
    x: *const f32, xi: i32, xj: i32, xk: i32, xl: i32,
    y: *const f32, yi: i32, yj: i32, yk: i32,
    axis1: i32, axis2: i32, axis3: i32,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size as usize) };
    minus::minus_with_d4_to_d3(x, xi as usize, xj as usize, xk as usize, xl as usize, y, yi as usize, yj as usize, yk as usize, axis1 as usize, axis2 as usize, axis3 as usize, result);
}
