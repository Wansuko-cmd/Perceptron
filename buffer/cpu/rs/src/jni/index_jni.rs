use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::jint;

use crate::index;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JIndex_gather(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    i: jint, j: jint, k: jint,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    index::gather(x, y, i as usize, j as usize, k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JIndex_scatterAdd(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    i: jint, j: jint, k: jint, b: jint,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };

    index::scatter_add(x, y, i as usize, j as usize, k as usize, b as usize, result);
}
