use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::jint;

use crate::core::shape;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_transposeD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::transpose_d2(x, xi as usize, xj as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_transposeD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint, xk: jint,
    axis_i: jint, axis_j: jint, axis_k: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::transpose_d3(
        x,
        xi as usize, xj as usize, xk as usize,
        axis_i as usize, axis_j as usize, axis_k as usize,
        result,
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_transposeD4(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint, xk: jint, xl: jint,
    axis_i: jint, axis_j: jint, axis_k: jint, axis_l: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::transpose_d4(
        x,
        xi as usize, xj as usize, xk as usize, xl as usize,
        axis_i as usize, axis_j as usize, axis_k as usize, axis_l as usize,
        result,
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_sliceD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    start: jint, end: jint, step: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::slice_d1(x, start as usize, end as usize, step as isize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_sliceD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint, axis: jint,
    start: jint, end: jint, step: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::slice_d2(x, xi as usize, xj as usize, axis as usize, start as usize, end as usize, step as isize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_sliceD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint, xk: jint, axis: jint,
    start: jint, end: jint, step: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::slice_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, start as usize, end as usize, step as isize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_copyIntoD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
    start: jint, end: jint, step: jint,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::copy_into_d1(x, result, start as usize, end as usize, step as isize);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_copyIntoD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
    ri: jint, rj: jint, axis: jint,
    start: jint, end: jint, step: jint,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::copy_into_d2(x, result, ri as usize, rj as usize, axis as usize, start as usize, end as usize, step as isize);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_shape_JShape_copyIntoD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
    ri: jint, rj: jint, rk: jint, axis: jint,
    start: jint, end: jint, step: jint,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    shape::copy_into_d3(x, result, ri as usize, rj as usize, rk as usize, axis as usize, start as usize, end as usize, step as isize);
}
