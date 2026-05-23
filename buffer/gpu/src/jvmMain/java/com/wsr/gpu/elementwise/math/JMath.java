package com.wsr.gpu.elementwise.math;

public class JMath {
    public native void exp(long x, long result, long runtime);
    public native void ln(long x, float e, long result, long runtime);
    public native void sigmoid(long x, long result, long runtime);
    public native void pow(long x, int n, long result, long runtime);
    public native void sqrt(long x, float e, long result, long runtime);
}
