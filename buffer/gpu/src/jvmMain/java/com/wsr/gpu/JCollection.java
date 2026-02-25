package com.wsr.gpu;

class JCollection {
    public native void averageD1(long x, long result, long context);
    public native void averageD2(long x, int xi, int xj, int axis, long result, long context);
    public native void averageD3(long x, int xi, int xj, int xk, int axis, long result, long context);
    public native void averageD4(long x, int xi, int xj, int xk, int xl, int axis, long result, long context);

    public native void maxD1(long x, long result, long context);
    public native void maxD2(long x, int xi, int xj, int axis, long result, long context);
    public native void maxD3(long x, int xi, int xj, int xk, int axis, long result, long context);
    public native void maxD4(long x, int xi, int xj, int xk, int xl, int axis, long result, long context);

    public native void minD1(long x, long result, long context);
    public native void minD2(long x, int xi, int xj, int axis, long result, long context);
    public native void minD3(long x, int xi, int xj, int xk, int axis, long result, long context);
    public native void minD4(long x, int xi, int xj, int xk, int xl, int axis, long result, long context);

    public native void sumD1(long x, long result, long context);
    public native void sumD2(long x, int xi, int xj, int axis, long result, long context);
    public native void sumD3(long x, int xi, int xj, int xk, int axis, long result, long context);
    public native void sumD4(long x, int xi, int xj, int xk, int xl, int axis, long result, long context);
}
