use jni::{JNIEnv};
use jni::objects::{JClass};
use jni::sys::{jint, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JShape_transposeD2(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    xi: jint,
    xj: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::shape::transpose_d2(x, xi as usize, xj as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JShape_transposeD3(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    xi: jint, xj: jint, xk: jint,
    axis_i: jint, axis_j: jint, axis_k: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::shape::transpose_d3(
        x,
        xi as usize, xj as usize, xk as usize,
        axis_i as usize, axis_j as usize, axis_k as usize,
        result,
        runtime,
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JShape_transposeD4(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    xi: jint, xj: jint, xk: jint, xl: jint,
    axis_i: jint, axis_j: jint, axis_k: jint, axis_l: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::shape::transpose_d4(
        x,
        xi as usize, xj as usize, xk as usize, xl as usize,
        axis_i as usize, axis_j as usize, axis_k as usize, axis_l as usize,
        result,
        runtime,
    );
}
