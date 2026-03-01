use jni::JNIEnv;
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat};

use crate::compare;
use crate::jni::utils::ByteBufferExt;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_greaterThanD1ToD0(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: jfloat,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::greater_than_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_greaterThanD1ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::greater_than_d1_to_d1(x, y, result);
}


#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_lessThanD1ToD0(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: jfloat,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::less_than_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_lessThanD1ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::less_than_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_equalsD1ToD0(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: jfloat,
    atol: jfloat,
    rtol: jfloat,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::equals_d1_to_d0(x, y, atol, rtol, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_equalsD1ToD1(
    env: JNIEnv,
    _class: JClass,
    x: JByteBuffer,
    y: JByteBuffer,
    atol: jfloat,
    rtol: jfloat,
    result: JByteBuffer,
) {    
    let x = unsafe { x.as_f32_slice(&env) };
    let y = unsafe { y.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::equals_d1_to_d1(x, y, atol, rtol, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_whereD0ToD0(
    env: JNIEnv,
    _class: JClass,
    condition: JByteBuffer,
    x: jfloat,
    y: jfloat,
    result: JByteBuffer,
) {
    let condition = unsafe { condition.as_f32_slice(&env) };
    let result = unsafe { result.as_f32_slice_mut(&env) };
    compare::where_d0_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_whereD0ToD1(
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
    compare::where_d0_to_d1(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_whereD1ToD0(
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
    compare::where_d1_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_cpu_JCompare_whereD1ToD1(
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
    compare::where_d1_to_d1(condition, x, y, result);
}
