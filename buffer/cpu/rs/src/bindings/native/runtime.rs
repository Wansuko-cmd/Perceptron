use crate::runtime::Runtime;
use crate::ops;

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_runtime_alloc(pool_size: i32) -> *mut Runtime {
    let runtime = ops::runtime::allocate(pool_size as u64);
    Box::into_raw(Box::new(runtime))
}

#[unsafe(no_mangle)]
pub extern "C" fn com_wsr_cpu_runtime_release(ptr: *mut Runtime) {
    unsafe { drop(Box::from_raw(ptr)) };
}
