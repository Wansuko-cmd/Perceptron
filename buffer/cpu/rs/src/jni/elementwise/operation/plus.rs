use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat, jint};

use crate::core::elementwise::operation::plus;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD0ToD1(
    env: JNIEnv,
    _class: JClass,
    x: jfloat,
    y: JByteBuffer,
    result: JByteBuffer,
) {
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d0_to_d1(x as f32, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD1ToD0(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: jfloat,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d1_to_d0(x, y as f32, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD1ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD1ToD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer, yi: jint, yj: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d1_to_d2(x, y, yi as usize, yj as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD1ToD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer, yi: jint, yj: jint, yk: jint,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d1_to_d3(x, y, yi as usize, yj as usize, yk as usize, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD2ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    y: JByteBuffer,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d2_to_d1(x, xi as usize, xj as usize, y, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD2ToD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint,
    y: JByteBuffer, yi: jint, yj: jint, yk: jint,
    axis1: jint, axis2: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d2_to_d3(x, xi as usize, xj as usize, y, yi as usize, yj as usize, yk as usize, axis1 as usize, axis2 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD3ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    y: JByteBuffer,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d3_to_d1(x, xi as usize, xj as usize, xk as usize, y, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD3ToD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    y: JByteBuffer, yi: jint, yj: jint,
    axis1: jint, axis2: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d3_to_d2(x, xi as usize, xj as usize, xk as usize, y, yi as usize, yj as usize, axis1 as usize, axis2 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD3ToD4(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint,
    y: JByteBuffer, yi: jint, yj: jint, yk: jint, yl: jint,
    axis1: jint, axis2: jint, axis3: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d3_to_d4(x, xi as usize, xj as usize, xk as usize, y, yi as usize, yj as usize, yk as usize, yl as usize, axis1 as usize, axis2 as usize, axis3 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD4ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint, xl: jint,
    y: JByteBuffer,
    axis: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d4_to_d1(x, xi as usize, xj as usize, xk as usize, xl as usize, y, axis as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD4ToD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint, xl: jint,
    y: JByteBuffer, yi: jint, yj: jint,
    axis1: jint, axis2: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d4_to_d2(x, xi as usize, xj as usize, xk as usize, xl as usize, y, yi as usize, yj as usize, axis1 as usize, axis2 as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_operation_plus_JPlus_plusD4ToD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer, xi: jint, xj: jint, xk: jint, xl: jint,
    y: JByteBuffer, yi: jint, yj: jint, yk: jint,
    axis1: jint, axis2: jint, axis3: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    plus::plus_with_d4_to_d3(x, xi as usize, xj as usize, xk as usize, xl as usize, y, yi as usize, yj as usize, yk as usize, axis1 as usize, axis2 as usize, axis3 as usize, result);
}
