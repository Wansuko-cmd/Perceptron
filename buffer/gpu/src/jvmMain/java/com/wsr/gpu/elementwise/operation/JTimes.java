package com.wsr.gpu.elementwise.operation;

public class JTimes {
    public native void timesD0ToD1(float x, long y, long result, long runtime);

    public native void timesD1ToD0(long x, float y, long result, long runtime);

    public native void timesD1ToD1(long x, long y, long result, long runtime);

    public native void timesD1ToD2(long x, long y, int yi, int yj, int axis, long result, long runtime);

    public native void timesD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long runtime);

    public native void timesD2ToD1(long x, int xi, int xj, long y, int axis, long result, long runtime);

    public native void timesD2ToD3(
            long x,
            int xi,
            int xj,
            long y,
            int yi,
            int yj,
            int yk,
            int axis1,
            int axis2,
            long result,
            long runtime
    );

    public native void timesD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long runtime);

    public native void timesD3ToD2(
            long x,
            int xi,
            int xj,
            int xk,
            long y,
            int yi,
            int yj,
            int axis1,
            int axis2,
            long result,
            long runtime
    );

    public native void timesD3ToD4(
            long x,
            int xi,
            int xj,
            int xk,
            long y,
            int yi,
            int yj,
            int yk,
            int yl,
            int axis1,
            int axis2,
            int axis3,
            long result,
            long runtime
    );

    public native void timesD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long runtime);

    public native void timesD4ToD2(
            long x,
            int xi,
            int xj,
            int xk,
            int xl,
            long y,
            int yi,
            int yj,
            int axis1,
            int axis2,
            long result,
            long runtime
    );

    public native void timesD4ToD3(
            long x,
            int xi,
            int xj,
            int xk,
            int xl,
            long y,
            int yi,
            int yj,
            int yk,
            int axis1,
            int axis2,
            int axis3,
            long result,
            long runtime
    );
}
