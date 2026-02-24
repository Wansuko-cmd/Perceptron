package com.wsr.gpu;

class JOperation {
    // 0次元
    public native void plusD0ToD1(float x, long y, long result, long context);

    // 1次元
    public native void plusD1ToD0(long x, float y, long result, long context);

    public native void plusD1ToD1(long x, long y, long result, long context);

    public native void plusD1ToD2(long x, long y, int yi, int yj, int axis, long result, long context);

    public native void plusD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long context);

    // 2次元
    public native void plusD2ToD1(long x, int xi, int xj, long y, int axis, long result, long context);

    public native void plusD2ToD3(
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
            long context
    );

    // 3次元
    public native void plusD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long context);

    public native void plusD3ToD2(
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
            long context
    );

    public native void plusD3ToD4(
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
            long context
    );

    // 4次元
    public native void plusD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long context);

    public native void plusD4ToD2(
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
            long context
    );

    public native void plusD4ToD3(
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
            long context
    );

    // 0次元
    public native void minusD0ToD1(float x, long y, long result, long context);

    // 1次元
    public native void minusD1ToD0(long x, float y, long result, long context);

    public native void minusD1ToD1(long x, long y, long result, long context);

    public native void minusD1ToD2(long x, long y, int yi, int yj, int axis, long result, long context);

    public native void minusD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long context);

    // 2次元
    public native void minusD2ToD1(long x, int xi, int xj, long y, int axis, long result, long context);

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
            long context
    );

    // 3次元
    public native void minusD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long context);

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
            long context
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
            long context
    );

    // 4次元
    public native void minusD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long context);

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
            long context
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
            long context
    );

    // 0次元
    public native void timesD0ToD1(float x, long y, long result, long context);

    // 1次元
    public native void timesD1ToD0(long x, float y, long result, long context);

    public native void timesD1ToD1(long x, long y, long result, long context);

    public native void timesD1ToD2(long x, long y, int yi, int yj, int axis, long result, long context);

    public native void timesD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long context);

    // 2次元
    public native void timesD2ToD1(long x, int xi, int xj, long y, int axis, long result, long context);

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
            long context
    );

    // 3次元
    public native void timesD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long context);

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
            long context
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
            long context
    );

    // 4次元
    public native void timesD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long context);

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
            long context
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
            long context
    );

    // 0次元
    public native void divD0ToD1(float x, long y, long result, long context);

    // 1次元
    public native void divD1ToD0(long x, float y, long result, long context);

    public native void divD1ToD1(long x, long y, long result, long context);

    public native void divD1ToD2(long x, long y, int yi, int yj, int axis, long result, long context);

    public native void divD1ToD3(long x, long y, int yi, int yj, int yk, int axis, long result, long context);

    // 2次元
    public native void divD2ToD1(long x, int xi, int xj, long y, int axis, long result, long context);

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
            long context
    );

    // 3次元
    public native void divD3ToD1(long x, int xi, int xj, int xk, long y, int axis, long result, long context);

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
            long context
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
            long context
    );

    // 4次元
    public native void divD4ToD1(long x, int xi, int xj, int xk, int xl, long y, int axis, long result, long context);

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
            long context
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
            long context
    );
}
