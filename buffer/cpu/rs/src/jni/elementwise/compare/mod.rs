pub mod r#where;

use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jlong};

use crate::core::elementwise::compare;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_greaterThanD1ToD0(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    compare::greater_than_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_greaterThanD1ToD1(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let y = unsafe { crate::jni::buffer::as_slice(y) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    compare::greater_than_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_lessThanD1ToD0(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    compare::less_than_d1_to_d0(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_lessThanD1ToD1(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let y = unsafe { crate::jni::buffer::as_slice(y) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    compare::less_than_d1_to_d1(x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_equalsD1ToD0(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    atol: jfloat,
    rtol: jfloat,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    compare::equals_d1_to_d0(x, y, atol, rtol, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_equalsD1ToD1(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    atol: jfloat,
    rtol: jfloat,
    result: jlong,
) {
    let x = unsafe { crate::jni::buffer::as_slice(x) };
    let y = unsafe { crate::jni::buffer::as_slice(y) };
    let result = unsafe { crate::jni::buffer::as_slice_mut(result) };
    compare::equals_d1_to_d1(x, y, atol, rtol, result);
}
