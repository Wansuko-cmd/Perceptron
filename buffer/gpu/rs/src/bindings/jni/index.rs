use jni::JNIEnv;
use jni::objects::{JClass};
use jni::sys::{jint, jlong};

use crate::ops;
use crate::resource::buffer::GPUBuffer;
use crate::runtime::Runtime;

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_wsr_gpu_JIndex_gather(
    _: JNIEnv,
    _class: JClass,
    x: jlong,
    y: jlong,
    i: jint, j: jint, k: jint,
    result: jlong,
    runtime: jlong,
 ) {    
    let x = unsafe { &*(x as *const GPUBuffer) };
    let y = unsafe { &*(y as *const GPUBuffer) };
    let result = unsafe { &*(result as *const GPUBuffer) };
    let runtime = unsafe { &mut *(runtime as *mut Runtime) };

    ops::index::gather(x, y, i as usize, j as usize, k as usize, result, runtime);
}
