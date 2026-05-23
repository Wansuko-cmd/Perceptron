package com.wsr.gpu.elementwise.compare;

public class JCompare {
    public native void greaterThanD1ToD0(long x, float y, long result, long runtime);
    public native void greaterThanD1ToD1(long x, long y, long result, long runtime);

    public native void lessThanD1ToD0(long x, float y, long result, long runtime);
    public native void lessThanD1ToD1(long x, long y, long result, long runtime);

    public native void equalsD1ToD0(long x, float y, float absoluteTolerance, float relativeTolerance, long result, long runtime);
    public native void equalsD1ToD1(long x, long y, float absoluteTolerance, float relativeTolerance, long result, long runtime);
}
