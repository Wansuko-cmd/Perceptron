package com.wsr.cpu.elementwise.operation.div;

import java.nio.ByteBuffer;

public class JDiv {
    // 0次元
    public native void divD0ToD1(float x, ByteBuffer y, ByteBuffer result);

    // 1次元
    public native void divD1ToD0(ByteBuffer x, float y, ByteBuffer result);

    public native void divD1ToD1(ByteBuffer x, ByteBuffer y, ByteBuffer result);

    public native void divD1ToD2(ByteBuffer x, ByteBuffer y, int yi, int yj, int axis, ByteBuffer result);

    public native void divD1ToD3(ByteBuffer x, ByteBuffer y, int yi, int yj, int yk, int axis, ByteBuffer result);

    // 2次元
    public native void divD2ToD1(ByteBuffer x, int xi, int xj, ByteBuffer y, int axis, ByteBuffer result);

    public native void divD2ToD3(
            ByteBuffer x,
            int xi,
            int xj,
            ByteBuffer y,
            int yi,
            int yj,
            int yk,
            int axis1,
            int axis2,
            ByteBuffer result
    );

    // 3次元
    public native void divD3ToD1(ByteBuffer x, int xi, int xj, int xk, ByteBuffer y, int axis, ByteBuffer result);

    public native void divD3ToD2(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            ByteBuffer y,
            int yi,
            int yj,
            int axis1,
            int axis2,
            ByteBuffer result
    );

    public native void divD3ToD4(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            ByteBuffer y,
            int yi,
            int yj,
            int yk,
            int yl,
            int axis1,
            int axis2,
            int axis3,
            ByteBuffer result
    );

    // 4次元
    public native void divD4ToD1(ByteBuffer x, int xi, int xj, int xk, int xl, ByteBuffer y, int axis, ByteBuffer result);

    public native void divD4ToD2(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            int xl,
            ByteBuffer y,
            int yi,
            int yj,
            int axis1,
            int axis2,
            ByteBuffer result
    );

    public native void divD4ToD3(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            int xl,
            ByteBuffer y,
            int yi,
            int yj,
            int yk,
            int axis1,
            int axis2,
            int axis3,
            ByteBuffer result
    );
}
