use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat, jint, jlong};

use crate::core::reduction;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_averageD1(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_averageD2(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_averageD3(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_maxD1(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_maxD2(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_maxD3(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_minD1(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_minD2(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_minD3(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_sumD1(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_sumD2(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_sumD3(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_maxIndexD1(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_maxIndexD2(
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
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_maxIndexD3(
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

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_topKD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    k: jint,
    seed: jlong,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::top_k_d1(x, k as usize, seed as u64) as f32;
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_topKD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    k: jint,
    axis: jint,
    seed: jlong,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::top_k_d2(x, xi as usize, xj as usize, k as usize, axis as usize, result, seed as u64);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_topKD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    k: jint,
    axis: jint,
    seed: jlong,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::top_k_d3(x, xi as usize, xj as usize, xk as usize, k as usize, axis as usize, result, seed as u64);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_topPD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    p: jfloat,
    seed: jlong,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    result[0] = reduction::top_p_d1(x, p as f32, seed as u64) as f32;
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_topPD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    p: jfloat,
    axis: jint,
    seed: jlong,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::top_p_d2(x, xi as usize, xj as usize, p as f32, axis as usize, result, seed as u64);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_reduction_JReduction_topPD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    p: jfloat,
    axis: jint,
    seed: jlong,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    reduction::top_p_d3(x, xi as usize, xj as usize, xk as usize, p as f32, axis as usize, result, seed as u64);
}
