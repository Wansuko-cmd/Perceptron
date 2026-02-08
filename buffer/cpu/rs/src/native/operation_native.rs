use crate::operation;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d0_to_d1(x: f32, y: *const f32, y_size: usize, result: *mut f32) {
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::plus_with_d0_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d1_to_d0(x: *const f32, x_size: usize, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d1_to_d1(x: *const f32, y: *const f32, size: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let y = unsafe { std::slice::from_raw_parts(y, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    operation::plus_with_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d1_to_d2(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::plus_with_d1_to_d2(x, y, yi, yj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d1_to_d3(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        2 => yk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::plus_with_d1_to_d3(x, y, yi, yj, yk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d2_to_d1(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d2_to_d1(x, xi, xj, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d2_to_d3(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::plus_with_d2_to_d3(x, xi, xj, y, yi, yj, yk, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d3_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d3_to_d1(x, xi, xj, xk, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d3_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d3_to_d2(x, xi, xj, xk, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d3_to_d4(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk * yl;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::plus_with_d3_to_d4(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d4_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        3 => xl,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d4_to_d1(x, xi, xj, xk, xl, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d4_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d4_to_d2(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_plus_d4_to_d3(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::plus_with_d4_to_d3(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d0_to_d1(x: f32, y: *const f32, y_size: usize, result: *mut f32) {
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::minus_with_d0_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d0(x: *const f32, x_size: usize, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d1(x: *const f32, y: *const f32, size: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let y = unsafe { std::slice::from_raw_parts(y, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    operation::minus_with_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d2(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::minus_with_d1_to_d2(x, y, yi, yj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d1_to_d3(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        2 => yk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::minus_with_d1_to_d3(x, y, yi, yj, yk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d2_to_d1(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d2_to_d1(x, xi, xj, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d2_to_d3(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::minus_with_d2_to_d3(x, xi, xj, y, yi, yj, yk, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d3_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d3_to_d1(x, xi, xj, xk, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d3_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d3_to_d2(x, xi, xj, xk, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d3_to_d4(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk * yl;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::minus_with_d3_to_d4(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d4_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        3 => xl,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d4_to_d1(x, xi, xj, xk, xl, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d4_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d4_to_d2(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_minus_d4_to_d3(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::minus_with_d4_to_d3(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d0_to_d1(x: f32, y: *const f32, y_size: usize, result: *mut f32) {
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::times_with_d0_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d1_to_d0(x: *const f32, x_size: usize, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d1_to_d1(x: *const f32, y: *const f32, size: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let y = unsafe { std::slice::from_raw_parts(y, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    operation::times_with_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d1_to_d2(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::times_with_d1_to_d2(x, y, yi, yj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d1_to_d3(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        2 => yk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::times_with_d1_to_d3(x, y, yi, yj, yk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d2_to_d1(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d2_to_d1(x, xi, xj, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d2_to_d3(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::times_with_d2_to_d3(x, xi, xj, y, yi, yj, yk, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d3_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d3_to_d1(x, xi, xj, xk, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d3_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d3_to_d2(x, xi, xj, xk, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d3_to_d4(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk * yl;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::times_with_d3_to_d4(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d4_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        3 => xl,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d4_to_d1(x, xi, xj, xk, xl, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d4_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d4_to_d2(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_times_d4_to_d3(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::times_with_d4_to_d3(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d0_to_d1(x: f32, y: *const f32, y_size: usize, result: *mut f32) {
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::div_with_d0_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d1_to_d0(x: *const f32, x_size: usize, y: f32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d1_to_d1(x: *const f32, y: *const f32, size: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    let y = unsafe { std::slice::from_raw_parts(y, size) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size) };
    operation::div_with_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d1_to_d2(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::div_with_d1_to_d2(x, y, yi, yj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d1_to_d3(
    x: *const f32,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: *mut f32,
) {
    let x_size = match axis {
        0 => yi,
        1 => yj,
        2 => yk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::div_with_d1_to_d3(x, y, yi, yj, yk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d2_to_d1(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d2_to_d1(x, xi, xj, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d2_to_d3(
    x: *const f32, xi: usize, xj: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::div_with_d2_to_d3(x, xi, xj, y, yi, yj, yk, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d3_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d3_to_d1(x, xi, xj, xk, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d3_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d3_to_d2(x, xi, xj, xk, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d3_to_d4(
    x: *const f32, xi: usize, xj: usize, xk: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk * yl;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, y_size) };
    operation::div_with_d3_to_d4(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d4_to_d1(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32,
    axis: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        3 => xl,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d4_to_d1(x, xi, xj, xk, xl, y, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d4_to_d2(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d4_to_d2(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_div_d4_to_d3(
    x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize,
    y: *const f32, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: *mut f32,
) {
    let x_size = xi * xj * xk * xl;
    let x = unsafe { std::slice::from_raw_parts(x, x_size) };

    let y_size = yi * yj * yk;
    let y = unsafe { std::slice::from_raw_parts(y, y_size) };

    let result = unsafe { std::slice::from_raw_parts_mut(result, x_size) };
    operation::div_with_d4_to_d3(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3, result);
}
