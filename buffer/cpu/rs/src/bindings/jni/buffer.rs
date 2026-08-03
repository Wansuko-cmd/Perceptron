use jni::JNIEnv;
use jni::objects::{JClass, JFloatArray, ReleaseMode};
use jni::sys::{jfloat, jfloatArray, jint, jlong};

use crate::resource::buffer::CPUBuffer;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_alloc(
    _env: JNIEnv,
    _class: JClass,
    size: jint,
) -> jlong {
    let buffer = CPUBuffer::create(size as usize);
    Box::into_raw(Box::new(buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_release(_env: JNIEnv, _class: JClass, ptr: jlong) {
    unsafe { drop(Box::from_raw(ptr as *mut CPUBuffer)) };
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_get(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    index: jint,
) -> jfloat {
    let buffer = unsafe { &*(ptr as *const CPUBuffer) };
    buffer[index as usize]
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_set(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    index: jint,
    value: jfloat,
) {
    let buffer = unsafe { &mut *(ptr as *mut CPUBuffer) };
    buffer[index as usize] = value;
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_readAll(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jfloatArray {
    let buffer = unsafe { &*(ptr as *const CPUBuffer) };
    let result = env.new_float_array(buffer.count() as i32).expect("failed to allocate result float[]");
    env.set_float_array_region(&result, 0, buffer)
        .expect("failed to copy into result float[]");
    result.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_writeAll(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    value: JFloatArray,
) {
    let elements = unsafe {
        env.get_array_elements(&value, ReleaseMode::NoCopyBack)
            .expect("failed to access source float[]")
    };
    let buffer = unsafe { &mut *(ptr as *mut CPUBuffer) };
    buffer.copy_from_slice(&elements);
}
