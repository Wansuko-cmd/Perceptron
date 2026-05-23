package com.wsr.gpu.elementwise.operation;

public class JMinus {
    public native void minusD0ToD1(float x, long y, long result, long runtime);

    public native void minusD1ToD0(long x, float y, long result, long runtime);

    public native void minusD1ToD1(long x, long y, long result, long runtime);

    public native void minusD1ToD2(long x, long y, int yi, int yj, int axis, long result, long runtime);

    public native void minusD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long runtime);

    public native void minusD2ToD1(long x, int xi, int xj, long y, int axis, long result, long runtime);

    public native void minusD2ToD3(
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

    public native void minusD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long runtime);

    public native void minusD3ToD2(
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

    public native void minusD3ToD4(
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

    public native void minusD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long runtime);

    public native void minusD4ToD2(
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

    public native void minusD4ToD3(
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
