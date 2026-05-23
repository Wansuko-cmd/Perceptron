package com.wsr.cpu.elementwise.compare.where;

import java.nio.ByteBuffer;

public class JWhere {
    public native void whereD0ToD0(ByteBuffer condition, float x, float y, ByteBuffer result);
    public native void whereD0ToD1(ByteBuffer condition, float x, ByteBuffer y, ByteBuffer result);
    public native void whereD1ToD0(ByteBuffer condition, ByteBuffer x, float y, ByteBuffer result);
    public native void whereD1ToD1(ByteBuffer condition, ByteBuffer x, ByteBuffer y, ByteBuffer result);
}
