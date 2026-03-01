package com.wsr.gpu;

public class JMath {
    public native void exp(long x, long result, long context);
    public native void ln(long x, float e, long result, long context);
    public native void sigmoid(long x, long result, long context);
    public native void pow(long x, int n, long result, long context);
    public native void sqrt(long x, float e, long result, long context);
}
