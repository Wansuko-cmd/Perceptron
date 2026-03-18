use crate::core::collection;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d1(x: *const f32, size: i32) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    return collection::average_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d2(x: *const f32, xi: i32, xj: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::average_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d3(x: *const f32, xi: i32, xj: i32, xk: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::average_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d1(x: *const f32, size: i32) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    return collection::max_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d2(x: *const f32, xi: i32, xj: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::max_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d3(x: *const f32, xi: i32, xj: i32, xk: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::max_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d1(x: *const f32, size: i32) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    return collection::min_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d2(x: *const f32, xi: i32, xj: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::min_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d3(x: *const f32, xi: i32, xj: i32, xk: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::min_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d1(x: *const f32, size: i32) -> f32 {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    return collection::sum_d1(x);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d2(x: *const f32, xi: i32, xj: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::sum_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d3(x: *const f32, xi: i32, xj: i32, xk: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };

    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    collection::sum_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}
