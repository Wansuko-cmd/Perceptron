use crate::ops::reduction;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_average_d1(x: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::average_d1(x);
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
    reduction::average_d2(x, xi as usize, xj as usize, axis as usize, result);
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
    reduction::average_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_d1(x: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::max_d1(x);
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
    reduction::max_d2(x, xi as usize, xj as usize, axis as usize, result);
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
    reduction::max_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_min_d1(x: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::min_d1(x);
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
    reduction::min_d2(x, xi as usize, xj as usize, axis as usize, result);
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
    reduction::min_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_sum_d1(x: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::sum_d1(x);
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
    reduction::sum_d2(x, xi as usize, xj as usize, axis as usize, result);
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
    reduction::sum_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_index_d1(x: *const f32, size: i32, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::max_index_d1(x) as f32;
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_index_d2(x: *const f32, xi: i32, xj: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    reduction::max_index_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_max_index_d3(x: *const f32, xi: i32, xj: i32, xk: i32, axis: i32, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    reduction::max_index_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_top_k_d1(x: *const f32, size: i32, k: i32, seed: i64, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::top_k_d1(x, k as usize, seed as u64) as f32;
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_top_k_d2(x: *const f32, xi: i32, xj: i32, k: i32, axis: i32, seed: i64, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    reduction::top_k_d2(x, xi as usize, xj as usize, k as usize, axis as usize, result, seed as u64);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_top_k_d3(x: *const f32, xi: i32, xj: i32, xk: i32, k: i32, axis: i32, seed: i64, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    reduction::top_k_d3(x, xi as usize, xj as usize, xk as usize, k as usize, axis as usize, result, seed as u64);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_top_p_d1(x: *const f32, size: i32, p: f32, seed: i64, result: *mut f32) {
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { &mut *result };
    *result = reduction::top_p_d1(x, p, seed as u64) as f32;
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_top_p_d2(x: *const f32, xi: i32, xj: i32, p: f32, axis: i32, seed: i64, result: *mut f32) {
    let x_size = xi * xj;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result_size = match axis {
        0 => xj,
        1 => xi,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    reduction::top_p_d2(x, xi as usize, xj as usize, p, axis as usize, result, seed as u64);
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_top_p_d3(x: *const f32, xi: i32, xj: i32, xk: i32, p: f32, axis: i32, seed: i64, result: *mut f32) {
    let x_size = xi * xj * xk;
    let x = unsafe { std::slice::from_raw_parts(x, x_size as usize) };
    let result_size = match axis {
        0 => xj * xk,
        1 => xi * xk,
        2 => xi * xj,
        _ => panic!("invalid parameter. [axis: {}]", axis)
    };
    let result = unsafe { std::slice::from_raw_parts_mut(result, result_size as usize) };
    reduction::top_p_d3(x, xi as usize, xj as usize, xk as usize, p, axis as usize, result, seed as u64);
}
