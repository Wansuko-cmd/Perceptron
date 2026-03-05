package com.wsr.gpu;

public class JRuntime {
    public native long allocate();
    public native void release(long ptr);
}
