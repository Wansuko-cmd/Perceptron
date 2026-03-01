use jni::{JNIEnv};
use jni::objects::{JClass};
use jni::sys::{jfloat, jint, jlong};

use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;
use crate::core::compare;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_greaterThanD1ToD0(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::gt_d1_to_d0(x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_greaterThanD1ToD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::gt_d1_to_d1(x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_lessThanD1ToD0(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::lt_d1_to_d0(x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_lessThanD1ToD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::lt_d1_to_d1(x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_equalsD1ToD0(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    atol: jfloat,
    rtol: jfloat,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::eq_d1_to_d0(x,  y, atol, rtol, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_equalsD1ToD1(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    atol: jfloat,
    rtol: jfloat,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::eq_d1_to_d1(x,  y, atol, rtol, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_whereD0ToD0(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jfloat,
    y: jfloat,
    result: jlong,
    context: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::where_d0_to_d0(condition, x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_whereD0ToD1(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jfloat,
    y: jlong,
    result: jlong,
    context: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::where_d0_to_d1(condition, x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_whereD1ToD0(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jlong,
    y: jfloat,
    result: jlong,
    context: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::where_d1_to_d0(condition, x,  y, result, context);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JCompare_whereD1ToD1(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jlong,
    y: jlong,
    result: jlong,
    context: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    compare::where_d1_to_d1(condition, x,  y, result, context);
}
