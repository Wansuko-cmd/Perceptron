use jni::JNIEnv;
use jni::objects::{JClass};
use jni::sys::{jboolean, jint, jlong};

use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;
use crate::core::mat_mul;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JMatMul_matMul(
    _: JNIEnv,
    _class: JClass,
    x: jlong, trans_x: jboolean,
    y: jlong, trans_y: jboolean,
    m: jint, n: jint, k: jint, b: jint,
    result: jlong,
    context: jlong,
 ) {    
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let context = unsafe { &*(context as *const Context) };

    mat_mul::mat_mul(
        x, trans_x != 0,
        y, trans_y != 0,
        m as usize, n as usize, k as usize, b as usize,
        result,
        context,
    );
}
