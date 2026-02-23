package com.wsr.gpu;

public class JContext {
    public native long allocate();
    public native void release(long ptr);
}
