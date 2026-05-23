package com.wsr.gpu.elementwise.operation;

public class JDiv {
    public native void divD0ToD1(float x, long y, long result, long runtime);

    public native void divD1ToD0(long x, float y, long result, long runtime);

    public native void divD1ToD1(long x, long y, long result, long runtime);

    public native void divD1ToD2(long x, long y, int yi, int yj, int axis, long result, long runtime);

    public native void divD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long runtime);

    public native void divD2ToD1(long x, int xi, int xj, long y, int axis, long result, long runtime);

    public native void divD2ToD3(
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

    public native void divD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long runtime);

    public native void divD3ToD2(
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

    public native void divD3ToD4(
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

    public native void divD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long runtime);

    public native void divD4ToD2(
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

    public native void divD4ToD3(
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
