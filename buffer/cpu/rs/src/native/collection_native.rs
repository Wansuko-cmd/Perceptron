use crate::collection;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d1(x: *const f32, size: usize) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    return collection::average_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d2(x: *const f32, xi: usize, xj: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::average_d2(x, xi, xj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d3(x: *const f32, xi: usize, xj: usize, xk: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::average_d3(x, xi, xj, xk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d4(x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk * xl) };

    let result_size = match axis {
        0 => xj * xk * xl,
        1 => xi * xk * xl,
        2 => xi * xj * xl,
        3 => xi * xj * xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::average_d4(x, xi, xj, xk, xl, axis, result);
}


#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d1(x: *const f32, size: usize) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    return collection::max_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d2(x: *const f32, xi: usize, xj: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::max_d2(x, xi, xj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d3(x: *const f32, xi: usize, xj: usize, xk: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::max_d3(x, xi, xj, xk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d4(x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk * xl) };

    let result_size = match axis {
        0 => xj * xk * xl,
        1 => xi * xk * xl,
        2 => xi * xj * xl,
        3 => xi * xj * xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::max_d4(x, xi, xj, xk, xl, axis, result);
}


#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d1(x: *const f32, size: usize) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    return collection::min_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d2(x: *const f32, xi: usize, xj: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::min_d2(x, xi, xj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d3(x: *const f32, xi: usize, xj: usize, xk: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::min_d3(x, xi, xj, xk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d4(x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk * xl) };

    let result_size = match axis {
        0 => xj * xk * xl,
        1 => xi * xk * xl,
        2 => xi * xj * xl,
        3 => xi * xj * xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::min_d4(x, xi, xj, xk, xl, axis, result);
}


#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d1(x: *const f32, size: usize) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size) };
    return collection::sum_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d2(x: *const f32, xi: usize, xj: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::sum_d2(x, xi, xj, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d3(x: *const f32, xi: usize, xj: usize, xk: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::sum_d3(x, xi, xj, xk, axis, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d4(x: *const f32, xi: usize, xj: usize, xk: usize, xl: usize, axis: usize, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, xi * xj * xk * xl) };

    let result_size = match axis {
        0 => xj * xk * xl,
        1 => xi * xk * xl,
        2 => xi * xj * xl,
        3 => xi * xj * xk,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size) };
    collection::sum_d4(x, xi, xj, xk, xl, axis, result);
}
