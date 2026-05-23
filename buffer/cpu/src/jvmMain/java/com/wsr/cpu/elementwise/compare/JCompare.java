package com.wsr.cpu.elementwise.compare;

import java.nio.ByteBuffer;

public class JCompare {
    public native void greaterThanD1ToD0(ByteBuffer x, float y, ByteBuffer result);
    public native void greaterThanD1ToD1(ByteBuffer x, ByteBuffer y, ByteBuffer result);

    public native void lessThanD1ToD0(ByteBuffer x, float y, ByteBuffer result);
    public native void lessThanD1ToD1(ByteBuffer x, ByteBuffer y, ByteBuffer result);

    public native void equalsD1ToD0(ByteBuffer x, float y, float absoluteTolerance, float relativeTolerance, ByteBuffer result);
    public native void equalsD1ToD1(ByteBuffer x, ByteBuffer y, float absoluteTolerance, float relativeTolerance, ByteBuffer result);
}
