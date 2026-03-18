package com.wsr.cpu;

import java.nio.ByteBuffer;

class JShape {
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
}
