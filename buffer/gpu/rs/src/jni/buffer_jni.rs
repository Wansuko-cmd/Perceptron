use jni::{JNIEnv};
use jni::objects::{JClass, JFloatArray, ReleaseMode};
use jni::sys::{jfloat, jfloatArray, jint, jlong};

use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_allocate(
    _: JNIEnv,
    _class: JClass,
    size: jint,
    context: jlong,
) -> jlong {
    let context = unsafe { &*(context as *const Context) };
    let buffer = GPUBuffer::create(size as usize, context);
    Box::into_raw(Box::new(buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_init(
    mut env: JNIEnv,
    _class: JClass,
    value: JFloatArray,
    context: jlong,
) -> jlong {
    let context = unsafe { &*(context as *const Context) };
    let value = unsafe { env.get_array_elements(&value, ReleaseMode::NoCopyBack).unwrap() };
    let buffer = GPUBuffer::init(&value, context);
    Box::into_raw(Box::new(buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_release(
    _: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    let _ = unsafe { Box::from_raw(ptr as *mut GPUBuffer) };
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_readAll(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    context_ptr: jlong,
) -> jfloatArray {
    let context = unsafe { &*(context_ptr as *const Context) };
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };

    let mut dest = vec![0.0f32; buffer.count()];
    buffer.read_all(&mut dest, context);

    let result = env.new_float_array(buffer.count() as i32).unwrap();
    let _ = env.set_float_array_region(&result, 0, &dest);
    result.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_write(
    _: JNIEnv,
    _class: JClass,
    ptr: jlong,
    index: jint,
    value: jfloat,
    context_ptr: jlong,
) {
    let context = unsafe { &*(context_ptr as *const Context) };
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };
    buffer.write(index as usize, value, context);
}
