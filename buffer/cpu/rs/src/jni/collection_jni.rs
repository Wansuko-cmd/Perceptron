use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat, jint};

use crate::core::collection;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_averageD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
) -> jfloat {    
    let x = unsafe { x.as_f32_slice(&env) };
    return collection::average_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_averageD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::average_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_averageD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::average_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_maxD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
) -> jfloat {    
    let x = unsafe { x.as_f32_slice(&env) };
    return collection::max_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_maxD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::max_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_maxD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::max_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_minD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
) -> jfloat {    
    let x = unsafe { x.as_f32_slice(&env) };
    return collection::min_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_minD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::min_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_minD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::min_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_sumD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
) -> jfloat {    
    let x = unsafe { x.as_f32_slice(&env) };
    return collection::sum_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_sumD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::sum_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCollection_sumD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    collection::sum_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}
