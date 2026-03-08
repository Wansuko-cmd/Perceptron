package com.wsr.gpu;

class JIndex {
    public native void gather(long x, long y, int i, int j, int k, long result, long runtime);
}
