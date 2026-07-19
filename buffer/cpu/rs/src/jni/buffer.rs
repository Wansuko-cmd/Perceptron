use jni::JNIEnv;
use jni::objects::{JClass, JFloatArray, ReleaseMode};
use jni::sys::{jfloat, jfloatArray, jint, jlong};

pub struct CPUBuffer {
    data: Box<[f32]>,
}

pub unsafe fn as_slice<'a>(ptr: jlong) -> &'a [f32] {
    unsafe { &(*(ptr as *const CPUBuffer)).data }
}

pub unsafe fn as_slice_mut<'a>(ptr: jlong) -> &'a mut [f32] {
    unsafe { &mut (*(ptr as *mut CPUBuffer)).data }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_alloc(
    _env: JNIEnv,
    _class: JClass,
    size: jint,
) -> jlong {
    let data = vec![0.0f32; size as usize].into_boxed_slice();
    Box::into_raw(Box::new(CPUBuffer { data })) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_release(_env: JNIEnv, _class: JClass, ptr: jlong) {
    unsafe { drop(Box::from_raw(ptr as *mut CPUBuffer)) };
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_get(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    index: jint,
) -> jfloat {
    unsafe { as_slice(ptr)[index as usize] }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_set(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    index: jint,
    value: jfloat,
) {
    unsafe { as_slice_mut(ptr)[index as usize] = value };
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_readAll(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jfloatArray {
    let slice = unsafe { as_slice(ptr) };
    let result = env.new_float_array(slice.len() as i32).expect("failed to allocate result float[]");
    env.set_float_array_region(&result, 0, slice)
        .expect("failed to copy into result float[]");
    result.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_knist_cpu_JBuffer_writeAll(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    value: JFloatArray,
) {
    let elements = unsafe {
        env.get_array_elements(&value, ReleaseMode::NoCopyBack)
            .expect("failed to access source float[]")
    };
    unsafe { as_slice_mut(ptr) }.copy_from_slice(&elements);
}
