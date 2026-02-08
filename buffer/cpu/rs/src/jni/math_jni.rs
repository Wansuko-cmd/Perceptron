use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat, jint};

use crate::math;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JMath_exp(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    math::exp_d1(x, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JMath_ln(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    e: jfloat,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    math::ln_d1(x, e as f32, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JMath_pow(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    n: jint,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    math::pow_d1(x, n as i32, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JMath_sqrt(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    e: jfloat,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    math::sqrt_d1(x, e as f32, result);
}
