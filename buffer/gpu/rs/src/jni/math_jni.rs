use jni::{JNIEnv};
use jni::objects::{JClass};
use jni::sys::{jlong};

use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;
use crate::core::math;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMath_exp(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    result: jlong,
    context: jlong,
) {
    let x = unsafe { &*(x as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    math::exp_d1(x, result, context);
}
