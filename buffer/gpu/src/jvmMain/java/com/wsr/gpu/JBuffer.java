package com.wsr.gpu;

public class JBuffer {
    public native long allocate(int size, long runtime);
    public native long init(float[] value, long runtime);
    public native void release(long ptr, long runtime);

    public native float[] readAll(long ptr, long runtime);
    public native void write(long ptr, int index, float value, long runtime);
}
