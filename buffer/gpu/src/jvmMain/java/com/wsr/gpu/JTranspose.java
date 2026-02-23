package com.wsr.gpu;

class JTranspose {
    public native void transposeD2(
            long x,
            int xi,
            int xj,
            long result,
            long context
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
            long context
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
            long context
    );
}
