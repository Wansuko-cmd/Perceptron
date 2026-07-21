@file:OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)

package com.wsr.knist.network

import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import kotlinx.serialization.KSerializer
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.cbor.Cbor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.okio.decodeFromBufferedSource
import kotlinx.serialization.json.okio.encodeToBufferedSink
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.plus
import kotlinx.serialization.modules.polymorphic
import kotlinx.serialization.modules.subclass
import okio.BufferedSink
import okio.BufferedSource

@Suppress("UNCHECKED_CAST")
class NetworkV2Serializer<I, O> : KSerializer<NetworkV2<I, O>> {
    private val idSerializer = GraphId.serializer()
    private val converterSerializer = PolymorphicSerializer(Converter::class) as KSerializer<Converter<Any?>>
    private val nodeSerializer = ListSerializer(PolymorphicSerializer(Graph.Node::class))
    private val outputSerializer = PolymorphicSerializer(Output::class)
    private val optimizerSerializer = PolymorphicSerializer(Optimizer::class)
    private val initializerSerializer = PolymorphicSerializer(WeightInitializer::class)

    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("com.wsr.knist.network.NetworkV2") {
            element("sourceId", idSerializer.descriptor)
            element("sourceConverter", converterSerializer.descriptor)
            element("graph", nodeSerializer.descriptor)
            element("sinkId", idSerializer.descriptor)
            element("sinkFrom", idSerializer.descriptor)
            element("sinkOutput", outputSerializer.descriptor)
            element("sinkConverter", converterSerializer.descriptor)
            element("optimizer", optimizerSerializer.descriptor)
            element("initializer", initializerSerializer.descriptor)
        }

    override fun serialize(encoder: Encoder, value: NetworkV2<I, O>) {
        encoder.encodeStructure(descriptor) {
            encodeSerializableElement(descriptor, 0, idSerializer, value.source.id)
            encodeSerializableElement(descriptor, 1, converterSerializer, value.source.converter as Converter<Any?>)
            encodeSerializableElement(descriptor, 2, nodeSerializer, value.graph)
            encodeSerializableElement(descriptor, 3, idSerializer, value.sink.id)
            encodeSerializableElement(descriptor, 4, idSerializer, value.sink.from)
            encodeSerializableElement(descriptor, 5, outputSerializer, value.sink.output)
            encodeSerializableElement(descriptor, 6, converterSerializer, value.sink.converter as Converter<Any?>)
            encodeSerializableElement(descriptor, 7, optimizerSerializer, value.optimizer)
            encodeSerializableElement(descriptor, 8, initializerSerializer, value.initializer)
        }
    }

    override fun deserialize(decoder: Decoder): NetworkV2<I, O> = decoder.decodeStructure(descriptor) {
        if (decodeSequentially()) {
            NetworkV2(
                source = Graph.Source(
                    id = decodeSerializableElement(descriptor, 0, idSerializer),
                    converter = decodeSerializableElement(descriptor, 1, converterSerializer) as Converter<I>,
                ),
                graph = decodeSerializableElement(descriptor, 2, nodeSerializer),
                sink = Graph.Sink(
                    id = decodeSerializableElement(descriptor, 3, idSerializer),
                    from = decodeSerializableElement(descriptor, 4, idSerializer),
                    output = decodeSerializableElement(descriptor, 5, outputSerializer),
                    converter = decodeSerializableElement(descriptor, 6, converterSerializer) as Converter<O>,
                ),
                optimizer = decodeSerializableElement(descriptor, 7, optimizerSerializer),
                initializer = decodeSerializableElement(descriptor, 8, initializerSerializer),
            )
        } else {
            var sourceId: GraphId? = null
            var sourceConverter: Converter<Any?>? = null
            var graph: List<Graph.Node>? = null
            var sinkId: GraphId? = null
            var sinkFrom: GraphId? = null
            var sinkOutput: Output? = null
            var sinkConverter: Converter<Any?>? = null
            var optimizer: Optimizer? = null
            var initializer: WeightInitializer? = null
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    0 -> sourceId = decodeSerializableElement(descriptor, 0, idSerializer)
                    1 -> sourceConverter = decodeSerializableElement(descriptor, 1, converterSerializer)
                    2 -> graph = decodeSerializableElement(descriptor, 2, nodeSerializer)
                    3 -> sinkId = decodeSerializableElement(descriptor, 3, idSerializer)
                    4 -> sinkFrom = decodeSerializableElement(descriptor, 4, idSerializer)
                    5 -> sinkOutput = decodeSerializableElement(descriptor, 5, outputSerializer)
                    6 -> sinkConverter = decodeSerializableElement(descriptor, 6, converterSerializer)
                    7 -> optimizer = decodeSerializableElement(descriptor, 7, optimizerSerializer)
                    8 -> initializer = decodeSerializableElement(descriptor, 8, initializerSerializer)
                    CompositeDecoder.DECODE_DONE -> break
                    else -> error("Unexpected index: $index")
                }
            }
            NetworkV2(
                source = Graph.Source(
                    id = checkNotNull(sourceId),
                    converter = checkNotNull(sourceConverter) as Converter<I>,
                ),
                graph = checkNotNull(graph),
                sink = Graph.Sink(
                    id = checkNotNull(sinkId),
                    from = checkNotNull(sinkFrom),
                    output = checkNotNull(sinkOutput),
                    converter = checkNotNull(sinkConverter) as Converter<O>,
                ),
                optimizer = checkNotNull(optimizer),
                initializer = checkNotNull(initializer),
            )
        }
    }

    companion object {
        fun <I, O> encodeToString(value: NetworkV2<I, O>) = json.encodeToString(
            serializer = NetworkV2Serializer(),
            value = value,
        )

        fun <I, O> encodeToBufferedSink(value: NetworkV2<I, O>, sink: BufferedSink) = json.encodeToBufferedSink(
            serializer = NetworkV2Serializer(),
            value = value,
            sink = sink,
        )

        fun <I, O> decodeFromString(value: String) = json.decodeFromString<NetworkV2<I, O>>(
            deserializer = NetworkV2Serializer(),
            string = value,
        )

        fun <I, O> decodeFromBufferedSource(source: BufferedSource) = json.decodeFromBufferedSource<NetworkV2<I, O>>(
            deserializer = NetworkV2Serializer(),
            source = source,
        )

        fun <I, O> encodeToCbor(value: NetworkV2<I, O>): ByteArray =
            cbor.encodeToByteArray(NetworkV2Serializer(), value)

        fun <I, O> encodeToCborSink(value: NetworkV2<I, O>, sink: BufferedSink) {
            sink.write(cbor.encodeToByteArray(NetworkV2Serializer<I, O>(), value))
        }

        fun <I, O> decodeFromCbor(bytes: ByteArray): NetworkV2<I, O> =
            cbor.decodeFromByteArray(NetworkV2Serializer(), bytes)

        fun <I, O> decodeFromCborSource(source: BufferedSource): NetworkV2<I, O> =
            cbor.decodeFromByteArray(NetworkV2Serializer(), source.readByteArray())

        private val modules
            get() = NetworkSerializer.modules.reduce { acc, module -> acc + module } + graphNodeSerializersModule

        private val json
            get() = Json { serializersModule = modules }

        private val cbor
            get() = Cbor { serializersModule = modules }
    }
}

private val graphNodeSerializersModule = SerializersModule {
    polymorphic(Graph.Node::class) {
        subclass(Graph.Node.Attach::class)
    }
}
