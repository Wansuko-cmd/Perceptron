package com.wsr.gpu;

class JMatMul {
    public native void matMul(
            long x,
            boolean transX,
            long y,
            boolean transY,
            int m,
            int n,
            int k,
            int b,
            long result,
            long runtime
    );
}
