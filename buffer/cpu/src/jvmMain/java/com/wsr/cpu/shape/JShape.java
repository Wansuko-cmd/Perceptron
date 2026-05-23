package com.wsr.cpu.shape;

import java.nio.ByteBuffer;

public class JShape {
    public native void transposeD2(
            ByteBuffer x,
            int xi,
            int xj,
            ByteBuffer result
    );

    public native void transposeD3(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            int axisI,
            int axisJ,
            int axisK,
            ByteBuffer result
    );

    public native void transposeD4(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            int xl,
            int axisI,
            int axisJ,
            int axisK,
            int axisL,
            ByteBuffer result
    );

    public native void sliceD1(
            ByteBuffer x,
            int start,
            int end,
            int step,
            ByteBuffer result
    );

    public native void sliceD2(
            ByteBuffer x,
            int xi,
            int xj,
            int axis,
            int start,
            int end,
            int step,
            ByteBuffer result
    );

    public native void sliceD3(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            int axis,
            int start,
            int end,
            int step,
            ByteBuffer result
    );

    public native void copyIntoD1(
            ByteBuffer x,
            ByteBuffer result,
            int start,
            int end,
            int step
    );

    public native void copyIntoD2(
            ByteBuffer x,
            ByteBuffer result,
            int ri,
            int rj,
            int axis,
            int start,
            int end,
            int step
    );

    public native void copyIntoD3(
            ByteBuffer x,
            ByteBuffer result,
            int ri,
            int rj,
            int rk,
            int axis,
            int start,
            int end,
            int step
    );

    public native void unfold(
            ByteBuffer x,
            int xi,
            int xj,
            int b,
            int window,
            int stride,
            int padding,
            ByteBuffer result
    );

    public native void fold(
            ByteBuffer x,
            int xi,
            int xj,
            int xk,
            int b,
            int stride,
            int padding,
            ByteBuffer result
    );
}
