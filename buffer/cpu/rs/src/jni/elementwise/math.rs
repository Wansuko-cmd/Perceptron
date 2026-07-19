use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jint, jlong};

use crate::core::elementwise::math;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_math_JMath_exp(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    math::exp_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_math_JMath_ln(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    e: jfloat,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    math::ln_d1(x, e as f32, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_math_JMath_sigmoid(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    math::sigmoid_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_math_JMath_pow(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    n: jint,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    math::pow_d1(x, n as i32, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_math_JMath_sqrt(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    e: jfloat,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    math::sqrt_d1(x, e as f32, result);
}
