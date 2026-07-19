use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jlong};

use crate::core::elementwise::generator;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_generator_JGenerator_random(
    _env: JNIEnv,
    _class: JClass,
    from: jfloat,
    until: jfloat,
    seed: jlong,
    result: jlong,
) {
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    generator::random_d1(from as f32, until as f32, seed as u64, result);
}
