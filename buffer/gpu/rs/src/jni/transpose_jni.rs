use jni::{JNIEnv};
use jni::objects::{JClass};
use jni::sys::{jint, jlong};

use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;
use crate::core::transpose;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JTranspose_transposeD2(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    xi: jint,
    xj: jint,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    transpose::transpose_d2(x, xi as usize, xj as usize, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JTranspose_transposeD3(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    xi: jint, xj: jint, xk: jint,
    axis_i: jint, axis_j: jint, axis_k: jint,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    transpose::transpose_d3(
        x,
        xi as usize, xj as usize, xk as usize,
        axis_i as usize, axis_j as usize, axis_k as usize,
        result,
        context,
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JTranspose_transposeD4(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    xi: jint, xj: jint, xk: jint, xl: jint,
    axis_i: jint, axis_j: jint, axis_k: jint, axis_l: jint,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    transpose::transpose_d4(
        x,
        xi as usize, xj as usize, xk as usize, xl as usize,
        axis_i as usize, axis_j as usize, axis_k as usize, axis_l as usize,
        result,
        context,
    );
}
