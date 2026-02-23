use jni::elements::ReleaseMode;
use jni::{Env, EnvUnowned};
use jni::objects::{JClass, JFloatArray};
use jni::sys::{jfloat, jfloatArray, jint, jlong};

use crate::buffer::GPUBuffer;
use crate::context::Context;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_allocate(
    _: EnvUnowned,
    _class: JClass,
    size: jint,
    context: jlong,
) -> jlong {
    let context = unsafe { &*(context as *const Context) };
    let buffer = GPUBuffer::create(size as usize, context);
    Box::into_raw(Box::new(buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_init(
    env: &mut Env,
    _class: JClass,
    value: JFloatArray,
    context: jlong,
) -> jlong {
    let context = unsafe { &*(context as *const Context) };
    let value = unsafe { value.get_elements(&env, ReleaseMode::NoCopyBack).unwrap() };
    let buffer = GPUBuffer::init(&value, context);
    Box::into_raw(Box::new(buffer)) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_release(
    _: EnvUnowned,
    _class: JClass,
    ptr: jlong,
) {
    let _ = unsafe { Box::from_raw(ptr as *mut GPUBuffer) };
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_readAll(
    env: &mut Env,
    _class: JClass,
    ptr: jlong,
    context_ptr: jlong,
) -> jfloatArray {
    let context = unsafe { &*(context_ptr as *const Context) };
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };

    let mut dest = vec![0.0f32; buffer.count()];
    buffer.read_all(&mut dest, context);

    let result = JFloatArray::new(env, buffer.count()).unwrap();
    let _ = JFloatArray::set_region(&result, &env, 0, &dest);
    result.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JBuffer_write(
    _: EnvUnowned,
    _class: JClass,
    ptr: jlong,
    index: jint,
    value: jfloat,
    context_ptr: jlong,
) {
    let context = unsafe { &*(context_ptr as *const Context) };
    let buffer = unsafe { &*(ptr as *const GPUBuffer) };
    buffer.write(index as usize, value, context);
}
