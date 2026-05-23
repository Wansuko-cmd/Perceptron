use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jint, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_averageD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::average_d1(x, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_averageD2(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::average_d2(x, xi as usize, xj as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_averageD3(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::average_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_maxD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::max_d1(x, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_maxD2(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::max_d2(x, xi as usize, xj as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_maxD3(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::max_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_minD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::min_d1(x, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_minD2(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::min_d2(x, xi as usize, xj as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_minD3(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::min_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_sumD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::sum_d1(x, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_sumD2(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::sum_d2(x, xi as usize, xj as usize, axis as usize, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_reduction_JCollection_sumD3(
    _: JNIEnv,
    _class: JClass,
    x: jlong, xi: jint, xj: jint, xk: jint,
    axis: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::collection::sum_d3(x, xi as usize, xj as usize, xk as usize, axis as usize, result, runtime);
}
