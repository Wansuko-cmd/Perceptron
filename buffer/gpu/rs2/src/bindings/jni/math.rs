use jni::{JNIEnv};
use jni::objects::{JClass};
use jni::sys::{jfloat, jint, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMath_exp(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::math::exp_d1(x, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMath_ln(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    e: jfloat,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::math::ln_d1(x, e as f32, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMath_sigmoid(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::math::sigmoid_d1(x, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMath_pow(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    n: jint,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::math::pow_d1(x, n as i32, result, runtime);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMath_sqrt(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    e: jfloat,
    result: jlong,
    runtime: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::math::sqrt_d1(x, e as f32, result, runtime);
}
