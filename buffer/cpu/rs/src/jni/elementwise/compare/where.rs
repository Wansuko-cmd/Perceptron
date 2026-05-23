use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::jfloat;

use crate::core::elementwise::compare::r#where;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_elementwise_compare_where_JWhere_whereD0ToD0(
    env: JNIEnv,
    _class: JClass,
    condition: JByteBuffer,
    x: jfloat,
    y: jfloat,
    result: JByteBuffer,
) {
    let condition = unsafe { condition.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    r#where::where_d0_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_elementwise_compare_where_JWhere_whereD0ToD1(
    env: JNIEnv,
    _class: JClass,
    condition: JByteBuffer,
    x: jfloat,
    y: JByteBuffer,
    result: JByteBuffer,
) {
    let condition = unsafe { condition.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    r#where::where_d0_to_d1(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_elementwise_compare_where_JWhere_whereD1ToD0(
    env: JNIEnv,
    _class: JClass,
    condition: JByteBuffer,
    x: JByteBuffer,
    y: jfloat,
    result: JByteBuffer,
) {
    let condition = unsafe { condition.as_f32_slice(&env) };
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    r#where::where_d1_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_elementwise_compare_where_JWhere_whereD1ToD1(
    env: JNIEnv,
    _class: JClass,
    condition: JByteBuffer,
    x: JByteBuffer,
    y: JByteBuffer,
    result: JByteBuffer,
) {
    let condition = unsafe { condition.as_f32_slice(&env) };
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    r#where::where_d1_to_d1(condition, x, y, result);
}
