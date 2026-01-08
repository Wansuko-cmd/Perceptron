package com.wsr.base.data

import kotlinx.serialization.KSerializer
import kotlinx.serialization.builtins.FloatArraySerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

object DataBufferSerializable : KSerializer<DataBuffer> {
    override val descriptor: SerialDescriptor = FloatArraySerializer().descriptor
    override fun serialize(encoder: Encoder, value: DataBuffer) {
        encoder.encodeSerializableValue(
            serializer = FloatArraySerializer(),
            value = value.toFloatArray(),
        )
    }

    override fun deserialize(decoder: Decoder): DataBuffer {
        val value = decoder.decodeSerializableValue(FloatArraySerializer())
        return DataBufferGenerator.create(value)
    }
}
