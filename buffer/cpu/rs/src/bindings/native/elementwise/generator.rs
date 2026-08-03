use crate::ops::elementwise::generator;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_random_d1(from: f32, until: f32, seed: i64, result: *mut f32, size: i32) {
    let result = unsafe { std::slice::from_raw_parts_mut(result, size as usize) };
    generator::random_d1(from, until, seed as u64, result);
}
