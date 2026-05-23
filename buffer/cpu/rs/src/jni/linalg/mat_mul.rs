use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jboolean, jint};

use crate::core::linalg::mat_mul;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_linalg_JMatMul_inner(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    b: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    mat_mul::inner(x, y, b as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_linalg_JMatMul_matMulD1ToD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    trans_y: jboolean,
    n: jint,
    k: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    mat_mul::mat_mul_d1_to_d2(x, y, trans_y != 0, n as usize, k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_linalg_JMatMul_matMulD2ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    trans_x: jboolean,
    y: JByteBuffer,
    m: jint,
    k: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    mat_mul::mat_mul_d2_to_d1(x, trans_x != 0, y, m as usize, k as usize, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_linalg_JMatMul_matMulD2ToD2(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    trans_x: jboolean,
    y: JByteBuffer,
    trans_y: jboolean,
    m: jint,
    k: jint,
    n: jint,
    b: jint,
    result: JByteBuffer,
) {
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    mat_mul::mat_mul_d2_to_d2(x, trans_x != 0, y, trans_y != 0, m as usize, k as usize, n as usize, b as usize, result);
}
