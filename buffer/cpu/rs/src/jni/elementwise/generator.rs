use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat, jlong};

use crate::core::elementwise::generator;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_generator_JGenerator_random(
    env: JNIEnv,
    _class: JClass,
    from: jfloat,
    until: jfloat,
    seed: jlong,
    result: JByteBuffer,
) {
    let result = unsafe { result.as_f32_slice_mut(&env) };
    generator::random_d1(from as f32, until as f32, seed as u64, result);
}
