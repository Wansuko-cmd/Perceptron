@file:OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)

package com.wsr.knist.network

import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import kotlinx.serialization.KSerializer
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure
import kotlinx.serialization.modules.plus

@Suppress("UNCHECKED_CAST")
private object GraphSourceSerializer : KSerializer<Graph.Source<*>> {
    private val idSerializer = GraphId.serializer()
    private val converterSerializer = PolymorphicSerializer(Converter::class) as KSerializer<Converter<Any?>>

    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("com.wsr.knist.network.Graph.Source") {
            element("id", idSerializer.descriptor)
            element("converter", converterSerializer.descriptor)
        }

    override fun serialize(encoder: Encoder, value: Graph.Source<*>) {
        encoder.encodeStructure(descriptor) {
            encodeSerializableElement(descriptor, 0, idSerializer, value.id)
            encodeSerializableElement(descriptor, 1, converterSerializer, value.converter as Converter<Any?>)
        }
    }

    override fun deserialize(decoder: Decoder): Graph.Source<*> = decoder.decodeStructure(descriptor) {
        if (decodeSequentially()) {
            Graph.Source(
                id = decodeSerializableElement(descriptor, 0, idSerializer),
                converter = decodeSerializableElement(descriptor, 1, converterSerializer),
            )
        } else {
            var id: GraphId? = null
            var converter: Converter<Any?>? = null
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    0 -> id = decodeSerializableElement(descriptor, 0, idSerializer)
                    1 -> converter = decodeSerializableElement(descriptor, 1, converterSerializer)
                    CompositeDecoder.DECODE_DONE -> break
                    else -> error("Unexpected index: $index")
                }
            }
            Graph.Source(id = checkNotNull(id), converter = checkNotNull(converter))
        }
    }
}

@Suppress("UNCHECKED_CAST")
private object GraphSinkSerializer : KSerializer<Graph.Sink<*>> {
    private val idSerializer = GraphId.serializer()
    private val converterSerializer = PolymorphicSerializer(Converter::class) as KSerializer<Converter<Any?>>
    private val outputSerializer = PolymorphicSerializer(Output::class)

    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("com.wsr.knist.network.Graph.Sink") {
            element("id", idSerializer.descriptor)
            element("from", idSerializer.descriptor)
            element("output", outputSerializer.descriptor)
            element("converter", converterSerializer.descriptor)
        }

    override fun serialize(encoder: Encoder, value: Graph.Sink<*>) {
        encoder.encodeStructure(descriptor) {
            encodeSerializableElement(descriptor, 0, idSerializer, value.id)
            encodeSerializableElement(descriptor, 1, idSerializer, value.from)
            encodeSerializableElement(descriptor, 2, outputSerializer, value.output)
            encodeSerializableElement(descriptor, 3, converterSerializer, value.converter as Converter<Any?>)
        }
    }

    override fun deserialize(decoder: Decoder): Graph.Sink<*> = decoder.decodeStructure(descriptor) {
        if (decodeSequentially()) {
            Graph.Sink(
                id = decodeSerializableElement(descriptor, 0, idSerializer),
                from = decodeSerializableElement(descriptor, 1, idSerializer),
                output = decodeSerializableElement(descriptor, 2, outputSerializer),
                converter = decodeSerializableElement(descriptor, 3, converterSerializer),
            )
        } else {
            var id: GraphId? = null
            var from: GraphId? = null
            var output: Output? = null
            var converter: Converter<Any?>? = null
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    0 -> id = decodeSerializableElement(descriptor, 0, idSerializer)
                    1 -> from = decodeSerializableElement(descriptor, 1, idSerializer)
                    2 -> output = decodeSerializableElement(descriptor, 2, outputSerializer)
                    3 -> converter = decodeSerializableElement(descriptor, 3, converterSerializer)
                    CompositeDecoder.DECODE_DONE -> break
                    else -> error("Unexpected index: $index")
                }
            }
            Graph.Sink(
                id = checkNotNull(id),
                from = checkNotNull(from),
                output = checkNotNull(output),
                converter = checkNotNull(converter),
            )
        }
    }
}

internal class GraphNetworkSerializer<T : GraphNetwork<T>>(
    private val factory: (
        sources: List<Graph.Source<*>>,
        graph: List<Graph.Node>,
        sinks: List<Graph.Sink<*>>,
        optimizer: Optimizer,
        initializer: WeightInitializer,
    ) -> T,
) : KSerializer<T> {
    private val sourceListSerializer = ListSerializer(GraphSourceSerializer)
    private val nodeSerializer = ListSerializer(PolymorphicSerializer(Graph.Node::class))
    private val sinkListSerializer = ListSerializer(GraphSinkSerializer)
    private val optimizerSerializer = PolymorphicSerializer(Optimizer::class)
    private val initializerSerializer = PolymorphicSerializer(WeightInitializer::class)

    override val descriptor: SerialDescriptor =
        buildClassSerialDescriptor("com.wsr.knist.network.GraphNetwork") {
            element("sources", sourceListSerializer.descriptor)
            element("graph", nodeSerializer.descriptor)
            element("sinks", sinkListSerializer.descriptor)
            element("optimizer", optimizerSerializer.descriptor)
            element("initializer", initializerSerializer.descriptor)
        }

    override fun serialize(encoder: Encoder, value: T) {
        encoder.encodeStructure(descriptor) {
            encodeSerializableElement(descriptor, 0, sourceListSerializer, value.sources)
            encodeSerializableElement(descriptor, 1, nodeSerializer, value.graph)
            encodeSerializableElement(descriptor, 2, sinkListSerializer, value.sinks)
            encodeSerializableElement(descriptor, 3, optimizerSerializer, value.optimizer)
            encodeSerializableElement(descriptor, 4, initializerSerializer, value.initializer)
        }
    }

    override fun deserialize(decoder: Decoder): T = decoder.decodeStructure(descriptor) {
        if (decodeSequentially()) {
            factory(
                decodeSerializableElement(descriptor, 0, sourceListSerializer),
                decodeSerializableElement(descriptor, 1, nodeSerializer),
                decodeSerializableElement(descriptor, 2, sinkListSerializer),
                decodeSerializableElement(descriptor, 3, optimizerSerializer),
                decodeSerializableElement(descriptor, 4, initializerSerializer),
            )
        } else {
            var sources: List<Graph.Source<*>>? = null
            var graph: List<Graph.Node>? = null
            var sinks: List<Graph.Sink<*>>? = null
            var optimizer: Optimizer? = null
            var initializer: WeightInitializer? = null
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    0 -> sources = decodeSerializableElement(descriptor, 0, sourceListSerializer)
                    1 -> graph = decodeSerializableElement(descriptor, 1, nodeSerializer)
                    2 -> sinks = decodeSerializableElement(descriptor, 2, sinkListSerializer)
                    3 -> optimizer = decodeSerializableElement(descriptor, 3, optimizerSerializer)
                    4 -> initializer = decodeSerializableElement(descriptor, 4, initializerSerializer)
                    CompositeDecoder.DECODE_DONE -> break
                    else -> error("Unexpected index: $index")
                }
            }
            factory(
                checkNotNull(sources),
                checkNotNull(graph),
                checkNotNull(sinks),
                checkNotNull(optimizer),
                checkNotNull(initializer),
            )
        }
    }
}
