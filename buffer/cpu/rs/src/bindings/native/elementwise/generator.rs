use crate::ops::elementwise::generator;
use crate::resource::buffer::CPUBuffer;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_random_d1(from: f32, until: f32, seed: i64, result: *mut CPUBuffer) {
    let result = unsafe { &mut *result };
    generator::random_d1(from, until, seed as u64, result);
}
