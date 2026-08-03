use crate::resource::buffer::CPUBuffer;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_buffer_alloc(size: i32) -> *mut CPUBuffer {
    Box::into_raw(Box::new(CPUBuffer::create(size as usize)))
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_buffer_init(value: *const f32, size: i32) -> *mut CPUBuffer {
    let value = unsafe { std::slice::from_raw_parts(value, size as usize) };
    Box::into_raw(Box::new(CPUBuffer::init(value)))
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_buffer_release(ptr: *mut CPUBuffer) {
    unsafe { drop(Box::from_raw(ptr)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_buffer_get(ptr: *const CPUBuffer, index: i32) -> f32 {
    let buffer = unsafe { &*ptr };
    buffer[index as usize]
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_buffer_set(ptr: *mut CPUBuffer, index: i32, value: f32) {
    let buffer = unsafe { &mut *ptr };
    buffer[index as usize] = value;
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_buffer_read_all(ptr: *const CPUBuffer, result: *mut f32) {
    let buffer = unsafe { &*ptr };
    let out = unsafe { std::slice::from_raw_parts_mut(result, buffer.count()) };
    out.copy_from_slice(buffer);
}
