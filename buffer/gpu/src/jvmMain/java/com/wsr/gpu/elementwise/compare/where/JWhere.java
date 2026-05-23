package com.wsr.gpu.elementwise.compare.where;

public class JWhere {
    public native void whereD0ToD0(long condition, float x, float y, long result, long runtime);
    public native void whereD0ToD1(long condition, float x, long y, long result, long runtime);
    public native void whereD1ToD0(long condition, long x, float y, long result, long runtime);
    public native void whereD1ToD1(long condition, long x, long y, long result, long runtime);
}
