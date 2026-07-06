use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::jlong;

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
