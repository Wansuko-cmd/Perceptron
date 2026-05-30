use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_elementwise_compare_where_JWhere_whereD0ToD0(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jfloat,
    y: jfloat,
    result: jlong,
    runtime: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::compare::where_d0_to_d0(condition, x, y, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_elementwise_compare_where_JWhere_whereD0ToD1(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jfloat,
    y: jlong,
    result: jlong,
    runtime: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::compare::where_d0_to_d1(condition, x, y, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_elementwise_compare_where_JWhere_whereD1ToD0(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jlong,
    y: jfloat,
    result: jlong,
    runtime: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::compare::where_d1_to_d0(condition, x, y, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_elementwise_compare_where_JWhere_whereD1ToD1(
    _: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jlong,
    y: jlong,
    result: jlong,
    runtime: jlong,
) {
    let condition = unsafe { &*(condition as *const GPUBuffer) };
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::compare::where_d1_to_d1(condition, x, y, result, runtime);
}
