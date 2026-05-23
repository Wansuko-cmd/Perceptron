package com.wsr.cpu.index;

import java.nio.ByteBuffer;

public class JIndex {
    public native void gather(ByteBuffer x, ByteBuffer y, int i, int j, int k, ByteBuffer result);
    public native void scatterAdd(ByteBuffer x, ByteBuffer y, int i, int j, int k, int b, ByteBuffer result);
}
