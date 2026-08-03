use crate::ops::elementwise::compare::r#where;

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_where_d0_to_d0(condition: *const f32, x: f32, y: f32, size: i32, result: *mut f32) {
    let condition = unsafe { std::slice::from_raw_parts(condition, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    r#where::where_d0_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_where_d0_to_d1(condition: *const f32, x: f32, y: *const f32, size: i32, result: *mut f32) {
    let condition = unsafe { std::slice::from_raw_parts(condition, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    r#where::where_d0_to_d1(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_where_d1_to_d0(condition: *const f32, x: *const f32, size: i32, y: f32, result: *mut f32) {
    let condition = unsafe { std::slice::from_raw_parts(condition, size as usize) };
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    r#where::where_d1_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn com_wsr_cpu_where_d1_to_d1(condition: *const f32, x: *const f32, y: *const f32, size: i32, result: *mut f32) {
    let condition = unsafe { std::slice::from_raw_parts(condition, size as usize) };
    let x = unsafe { std::slice::from_raw_parts(x, size as usize) };
    let y = unsafe { std::slice::from_raw_parts(y, size as usize) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    r#where::where_d1_to_d1(condition, x, y, result);
}
