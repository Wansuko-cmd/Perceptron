package com.wsr.gpu;

public class JBuffer {
    public native long allocate(int size, long context);
    public native long init(float[] value, long context);
    public native void release(long ptr, long context);

    public native float[] readAll(long ptr, long context);
    public native void write(long ptr, int index, float value, long context);
}
