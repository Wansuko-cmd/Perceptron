use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::jint;

use crate::core::reduction;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_averageD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::average_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_averageD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::average_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_averageD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::average_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_maxD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::max_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_maxD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::max_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_maxD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::max_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_minD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::min_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_minD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::min_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_minD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::min_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_sumD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::sum_d1(x);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_sumD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::sum_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_sumD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::sum_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_maxIndexD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::max_index_d1(x) as f32;
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_maxIndexD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::max_index_d2(x, xi as usize, xj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JCollection_maxIndexD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::max_index_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result);
}
