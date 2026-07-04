use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_gpu_elementwise_generator_JGenerator_random(
    _: JNIEnv,
    _class: JClass,
    from: jfloat,
    until: jfloat,
    seed: jlong,
    result: jlong,
    runtime: jlong,
) {
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::elementwise::generator::random_d1(from as f32, until as f32, seed as u32, result, runtime);
}
