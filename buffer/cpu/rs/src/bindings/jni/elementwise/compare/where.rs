use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jfloat, jlong};

use crate::ops::elementwise::compare::r#where;
use crate::resource::buffer::CPUBuffer;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_where_JWhere_whereD0ToD0(
    _env: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jfloat,
    y: jfloat,
    result: jlong,
) {
    let condition = unsafe { &*(condition as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
    r#where::where_d0_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_where_JWhere_whereD0ToD1(
    _env: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jfloat,
    y: jlong,
    result: jlong,
) {
    let condition = unsafe { &*(condition as *const CPUBuffer) };
    let y = unsafe { &*(y as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
    r#where::where_d0_to_d1(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_where_JWhere_whereD1ToD0(
    _env: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jlong,
    y: jfloat,
    result: jlong,
) {
    let condition = unsafe { &*(condition as *const CPUBuffer) };
    let x = unsafe { &*(x as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
    r#where::where_d1_to_d0(condition, x, y, result);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_elementwise_compare_where_JWhere_whereD1ToD1(
    _env: JNIEnv,
    _class: JClass,
    condition: jlong,
    x: jlong,
    y: jlong,
    result: jlong,
) {
    let condition = unsafe { &*(condition as *const CPUBuffer) };
    let x = unsafe { &*(x as *const CPUBuffer) };
    let y = unsafe { &*(y as *const CPUBuffer) };
    let result = unsafe { &mut *(result as *mut CPUBuffer) };
    r#where::where_d1_to_d1(condition, x, y, result);
}
