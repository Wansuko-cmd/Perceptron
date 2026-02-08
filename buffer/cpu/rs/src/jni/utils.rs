use jni::JNIEnv;
use jni::objects::{JByteBuffer};

pub trait ByteBufferExt<'a> {
    unsafe fn as_f32_slice(self, env: &JNIEnv<'a>) -> &'a [f32];
    unsafe fn as_f32_slice_mut(self, env: &JNIEnv<'a>) -> &'a mut [f32];
}

impl<'a> ByteBufferExt<'a> for &JByteBuffer<'a> {
    unsafe fn as_f32_slice(self, env: &JNIEnv<'a>) -> &'a [f32] {
        let addr = env.get_direct_buffer_address(self).unwrap();
        let cap = env.get_direct_buffer_capacity(self).unwrap();
        return unsafe { std::slice::from_raw_parts(addr as *const f32, cap / 4) };
    }

    unsafe fn as_f32_slice_mut(self, env: &JNIEnv<'a>) -> &'a mut [f32] {
        let addr = env.get_direct_buffer_address(self).unwrap();
        let cap = env.get_direct_buffer_capacity(self).unwrap();
        return unsafe { std::slice::from_raw_parts_mut(addr as *mut f32, cap / 4) };
    }
}
