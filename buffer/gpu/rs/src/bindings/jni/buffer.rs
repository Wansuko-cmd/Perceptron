use jni::{JNIEnv};
use jni::objects::{JClass, JFloatArray, ReleaseMode};
use jni::sys::{jboolean, jfloat, jfloatArray, jint, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_allocate(
    _: JNIEnv,
    _class: JClass,
    size: jint,
    runtime: jlong,
) -> jlong {
    let runtime = unsafe { &*(runtime as *const Runtime) };
    let buffer = ops::buffer::create(size as usize, runtime);
    Box::into_raw(Box::new(buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_init(
    mut env: JNIEnv,
    _class: JClass,
    value: JFloatArray,
    runtime: jlong,
) -> jlong {
    let value = unsafe { env.get_array_elements(&value, ReleaseMode::NoCopyBack).unwrap() };
    let runtime = unsafe { &*(runtime as *const Runtime) };
    let buffer = ops::buffer::init(&value, runtime);
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
    runtime_ptr: jlong,
) -> jfloatArray {
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime_ptr as *mut Runtime) };

    let buffer = ops::buffer::read_all(buffer, runtime);

    let result = env.new_float_array(buffer.len() as i32).unwrap();
    let _ = env.set_float_array_region(&result, 0, &buffer);
    result.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_write(
    _: JNIEnv,
    _class: JClass,
    ptr: jlong,
    index: jint,
    value: jfloat,
    runtime_ptr: jlong,
) {
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime_ptr as *mut Runtime) };
    ops::buffer::write(buffer, index as usize, value, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_slice(
    _: JNIEnv,
    _: JClass,
    ptr: jlong,
    start: jint,
    end: jint,
    step: jint,
    runtime_ptr: jlong,
) -> jlong {
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime_ptr as *mut Runtime) };

    let result = ops::buffer::slice(buffer, start as usize, end as usize, step as isize, runtime);
    Box::into_raw(Box::new(result)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_copyInto(
    _: JNIEnv,
    _: JClass,
    ptr: jlong,
    dest_ptr: jlong,
    dest_start: jint,
    dest_end: jint,
    dest_step: jint,
    runtime_ptr: jlong,
) {
    let src = unsafe { &*(ptr as *const GPUBuffer) };
    let dest = unsafe { &*(dest_ptr as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime_ptr as *mut Runtime) };

    ops::buffer::copy_into(src, dest, dest_start as usize, dest_end as usize, dest_step as isize, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_contentEquals(
    _: JNIEnv,
    _: JClass,
    ptr: jlong,
    other_ptr: jlong,
    runtime_ptr: jlong,
) -> jboolean {
    let x = unsafe { &*(ptr as *const GPUBuffer) };
    let y = unsafe { &*(other_ptr as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime_ptr as *mut Runtime) };

    if ops::buffer::content_equals(x, y, runtime) { 1 } else { 0 }
}
