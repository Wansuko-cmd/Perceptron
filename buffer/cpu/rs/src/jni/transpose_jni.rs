use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jint};

use crate::transpose;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JTranspose_transposeD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    transpose::transpose_d2(x, xi as usize, xj as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JTranspose_transposeD3(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint, xk: jint,
    axis_i: jint, axis_j: jint, axis_k: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    transpose::transpose_d3(
        x,
        xi as usize, xj as usize, xk as usize,
        axis_i as usize, axis_j as usize, axis_k as usize,
        result,
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JTranspose_transposeD4(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    xi: jint, xj: jint, xk: jint, xl: jint,
    axis_i: jint, axis_j: jint, axis_k: jint, axis_l: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    transpose::transpose_d4(
        x,
        xi as usize, xj as usize, xk as usize, xl as usize,
        axis_i as usize, axis_j as usize, axis_k as usize, axis_l as usize,
        result,
    );
}
