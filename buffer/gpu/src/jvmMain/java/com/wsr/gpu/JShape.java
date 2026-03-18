package com.wsr.gpu;

class JShape {
    public native void transposeD2(
            long x,
            int xi,
            int xj,
            long result,
            long runtime
    );

    public native void transposeD3(
            long x,
            int xi,
            int xj,
            int xk,
            int axisI,
            int axisJ,
            int axisK,
            long result,
            long runtime
    );

    public native void transposeD4(
            long x,
            int xi,
            int xj,
            int xk,
            int xl,
            int axisI,
            int axisJ,
            int axisK,
            int axisL,
            long result,
            long runtime
    );

    public native void sliceD1(
            long x,
            int start,
            int end,
            int step,
            long result,
            long runtime
    );

    public native void sliceD2(
            long x,
            int xi,
            int xj,
            int axis,
            int start,
            int end,
            int step,
            long result,
            long runtime
    );

    public native void sliceD3(
            long x,
            int xi,
            int xj,
            int xk,
            int axis,
            int start,
            int end,
            int step,
            long result,
            long runtime
    );
}
