use crate::ops::index;

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_gather(
    x: *const f32, y: *const f32,
    i: i32, j: i32, k: i32, n: i32,
    result: *mut f32,
) {
    let x = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, (i * j * k) as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, (i * n * k) as usize) };
    index::gather(x, y, i as usize, j as usize, k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_scatter_add(
    x: *const f32, y: *const f32,
    i: i32, j: i32, k: i32, n: i32, b: i32,
    result: *mut f32,
) {
    let x = unsafe { std::slice::from_raw_parts(x, (i * n * k) as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, n as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, (i * j * k) as usize) };
    index::scatter_add(x, y, i as usize, j as usize, k as usize, b as usize, result);
}
