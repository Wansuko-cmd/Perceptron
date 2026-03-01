package com.wsr.gpu;

public class JCompare {
    public native void greaterThanD1ToD0(long x, float y, long result, long context);
    public native void greaterThanD1ToD1(long x, long y, long result, long context);

    public native void lessThanD1ToD0(long x, float y, long result, long context);
    public native void lessThanD1ToD1(long x, long y, long result, long context);

    public native void equalsD1ToD0(long x, float y, float absoluteTolerance, float relativeTolerance, long result, long context);
    public native void equalsD1ToD1(long x, long y, float absoluteTolerance, float relativeTolerance, long result, long context);

    public native void whereD0ToD0(long condition, float x, float y, long result, long context);
    public native void whereD0ToD1(long condition, float x, long y, long result, long context);
    public native void whereD1ToD0(long condition, long x, float y, long result, long context);
    public native void whereD1ToD1(long condition, long x, long y, long result, long context);
}
