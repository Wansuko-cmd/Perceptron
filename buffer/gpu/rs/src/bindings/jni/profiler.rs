use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jlong, jlongArray};

use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_JProfiler_takeCPU(
    _: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
) -> jlong {
    let runtime = unsafe { &*(runtime_ptr as *const Runtime) };
    runtime.cpu_profiler.take() as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_JProfiler_takeGPU(
    env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
) -> jlongArray {
    let runtime = unsafe { &*(runtime_ptr as *const Runtime) };
    let result = runtime.gpu_profiler.take(&runtime.device, &runtime.queue);

    let values: [i64; 2] = [result.gpu_busy_ns as i64, result.gpu_span_ns as i64];
    let array = env.new_long_array(values.len() as i32).unwrap();
    let _ = env.set_long_array_region(&array, 0, &values);
    array.into_raw()
}
