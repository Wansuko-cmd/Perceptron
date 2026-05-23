package com.wsr.cpu.elementwise.operation.times;

import java.nio.ByteBuffer;

public class JTimes {
    // 0次元
    public native void timesD0ToD1(float x, ByteBuffer y, ByteBuffer result);

    // 1次元
    public native void timesD1ToD0(ByteBuffer x, float y, ByteBuffer result);

    public native void timesD1ToD1(ByteBuffer x, ByteBuffer y, ByteBuffer result);

    public native void timesD1ToD2(ByteBuffer x, ByteBuffer y, int yi, int yj, int axis, ByteBuffer result);

    public native void timesD1ToD3(ByteBuffer x, ByteBuffer y, int yi, int yj, int yk, int axis, ByteBuffer result);

    // 2次元
    public native void timesD2ToD1(ByteBuffer x, int xi, int xj, ByteBuffer y, int axis, ByteBuffer result);

    public native void timesD2ToD3(
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
    public native void timesD3ToD1(ByteBuffer x, int xi, int xj, int xk, ByteBuffer y, int axis, ByteBuffer result);

    public native void timesD3ToD2(
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

    public native void timesD3ToD4(
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
    public native void timesD4ToD1(ByteBuffer x, int xi, int xj, int xk, int xl, ByteBuffer y, int axis, ByteBuffer result);

    public native void timesD4ToD2(
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

    public native void timesD4ToD3(
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
