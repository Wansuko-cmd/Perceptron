use jni::{JNIEnv};
use jni::objects::{JClass, JFloatArray, ReleaseMode};
use jni::sys::{jboolean, jfloat, jfloatArray, jint, jlong};

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

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_slice(
    _: JNIEnv,
    _: JClass,
    ptr: jlong,
    start: jint,
    end: jint,
    context_ptr: jlong,
) -> jlong {
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };
    let context = unsafe { &*(context_ptr as *const Context) };

    let sliced_buffer = buffer.slice(start as usize, end as usize, context);
    Box::into_raw(Box::new(sliced_buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_copyInto(
    _: JNIEnv,
    _: JClass,
    ptr: jlong,
    dest_ptr: jlong,
    destination_offset: jint,
    context_ptr: jlong,
) {
    let src_buffer = unsafe { &*(ptr as *const GPUBuffer) };
    let dest_buffer = unsafe { &*(dest_ptr as *const GPUBuffer) };
    let context = unsafe { &*(context_ptr as *const Context) };

    src_buffer.copy_into(dest_buffer, destination_offset as usize, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_contentEquals(
    _: JNIEnv,
    _: JClass,
    ptr: jlong,
    other_ptr: jlong,
    context_ptr: jlong,
) -> jboolean {
    let self_buffer = unsafe { &*(ptr as *const GPUBuffer) };
    let other_buffer = unsafe { &*(other_ptr as *const GPUBuffer) };
    let context = unsafe { &*(context_ptr as *const Context) };

    return if self_buffer.content_equals(other_buffer, context) { 1 } else { 0 };
}
