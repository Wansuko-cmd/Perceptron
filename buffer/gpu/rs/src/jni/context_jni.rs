use jni::{EnvUnowned};
use jni::objects::{JClass};
use jni::sys::jlong;

use crate::context;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JContext_allocate(
    _: EnvUnowned,
    _class: JClass,
) -> jlong {
    let context = pollster::block_on(context::Context::new());
    Box::into_raw(Box::new(context)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JContext_release(
    _: EnvUnowned,
    _class: JClass,
    ptr: jlong,
) {
    let _ = unsafe { Box::from_raw(ptr as *mut context::Context) };
}
