pub mod r#where;

use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jlong};

use crate::ops::elementwise::compare;
use crate::resource::buffer::CPUBuffer;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_JCompare_greaterThanD1ToD0(
    _env: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jfloat,
    result: jlong,
) {
    let x = unsafe { &*(x as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
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
    let x = unsafe { &*(x as *const CPUBuffer) };
    let y = unsafe { &*(y as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
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
    let x = unsafe { &*(x as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
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
    let x = unsafe { &*(x as *const CPUBuffer) };
    let y = unsafe { &*(y as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
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
    let x = unsafe { &*(x as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
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
    let x = unsafe { &*(x as *const CPUBuffer) };
    let y = unsafe { &*(y as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
    compare::equals_d1_to_d1(x, y, atol, rtol, result);
}
