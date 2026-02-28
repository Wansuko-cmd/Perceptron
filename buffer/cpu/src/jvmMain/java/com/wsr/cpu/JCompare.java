package com.wsr.cpu;

import java.nio.ByteBuffer;

public class JCompare {
    public native void greaterThanD1ToD0(ByteBuffer x, float y, ByteBuffer result);
    public native void greaterThanD1ToD1(ByteBuffer x, ByteBuffer y, ByteBuffer result);

    public native void lessThanD1ToD0(ByteBuffer x, float y, ByteBuffer result);
    public native void lessThanD1ToD1(ByteBuffer x, ByteBuffer y, ByteBuffer result);

    public native void whereD0ToD1(ByteBuffer condition, float x, ByteBuffer y, ByteBuffer result);
    public native void whereD1ToD0(ByteBuffer condition, ByteBuffer x, float y, ByteBuffer result);
    public native void whereD1ToD1(ByteBuffer condition, ByteBuffer x, ByteBuffer y, ByteBuffer result);
}
