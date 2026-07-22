@file:OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)

package com.wsr.knist.network

import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.converter.raw.RawD1
import com.wsr.knist.network.converter.raw.RawD2
import com.wsr.knist.network.converter.raw.RawD3
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.initializer.He
import com.wsr.knist.network.initializer.Random
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.initializer.Xavier
import com.wsr.knist.network.join.Join
import com.wsr.knist.network.join.add.AddD1
import com.wsr.knist.network.join.add.AddD2
import com.wsr.knist.network.join.add.AddD3
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.adam.Adam
import com.wsr.knist.network.optimizer.adam.AdamD1
import com.wsr.knist.network.optimizer.adam.AdamD2
import com.wsr.knist.network.optimizer.adam.AdamD3
import com.wsr.knist.network.optimizer.adam.AdamD4
import com.wsr.knist.network.optimizer.adam.AdamW
import com.wsr.knist.network.optimizer.adam.AdamWD1
import com.wsr.knist.network.optimizer.adam.AdamWD2
import com.wsr.knist.network.optimizer.adam.AdamWD3
import com.wsr.knist.network.optimizer.adam.AdamWD4
import com.wsr.knist.network.optimizer.freeze.Freeze
import com.wsr.knist.network.optimizer.freeze.FreezeD1
import com.wsr.knist.network.optimizer.freeze.FreezeD2
import com.wsr.knist.network.optimizer.freeze.FreezeD3
import com.wsr.knist.network.optimizer.freeze.FreezeD4
import com.wsr.knist.network.optimizer.momentum.Momentum
import com.wsr.knist.network.optimizer.momentum.MomentumD1
import com.wsr.knist.network.optimizer.momentum.MomentumD2
import com.wsr.knist.network.optimizer.momentum.MomentumD3
import com.wsr.knist.network.optimizer.momentum.MomentumD4
import com.wsr.knist.network.optimizer.rms.RmsProp
import com.wsr.knist.network.optimizer.rms.RmsPropD1
import com.wsr.knist.network.optimizer.rms.RmsPropD2
import com.wsr.knist.network.optimizer.rms.RmsPropD3
import com.wsr.knist.network.optimizer.rms.RmsPropD4
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.optimizer.sgd.SgdD1
import com.wsr.knist.network.optimizer.sgd.SgdD2
import com.wsr.knist.network.optimizer.sgd.SgdD3
import com.wsr.knist.network.optimizer.sgd.SgdD4
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.output.mean.MeanSquareD1
import com.wsr.knist.network.output.mean.MeanSquareD2
import com.wsr.knist.network.output.sigmoid.SigmoidWithLossD1
import com.wsr.knist.network.output.sigmoid.SigmoidWithLossD2
import com.wsr.knist.network.output.softmax.SoftmaxWithLossD1
import com.wsr.knist.network.output.softmax.SoftmaxWithLossD2
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.compute.affine.AffineD1
import com.wsr.knist.network.process.compute.affine.AffineD2
import com.wsr.knist.network.process.compute.attention.AttentionD2
import com.wsr.knist.network.process.compute.attention.bias.AttentionBiasD2
import com.wsr.knist.network.process.compute.bias.d1.BiasD1
import com.wsr.knist.network.process.compute.bias.d2.BiasAxisD2
import com.wsr.knist.network.process.compute.bias.d2.BiasD2
import com.wsr.knist.network.process.compute.bias.d3.BiasAxisD3
import com.wsr.knist.network.process.compute.bias.d3.BiasD3
import com.wsr.knist.network.process.compute.conv.ConvD1
import com.wsr.knist.network.process.compute.conv.ConvD2
import com.wsr.knist.network.process.compute.debug.DebugD1
import com.wsr.knist.network.process.compute.debug.DebugD2
import com.wsr.knist.network.process.compute.debug.DebugD3
import com.wsr.knist.network.process.compute.dropout.DropoutD1
import com.wsr.knist.network.process.compute.dropout.DropoutD2
import com.wsr.knist.network.process.compute.dropout.DropoutD3
import com.wsr.knist.network.process.compute.function.relu.LeakyReLUD1
import com.wsr.knist.network.process.compute.function.relu.LeakyReLUD2
import com.wsr.knist.network.process.compute.function.relu.LeakyReLUD3
import com.wsr.knist.network.process.compute.function.relu.ReLUD1
import com.wsr.knist.network.process.compute.function.relu.ReLUD2
import com.wsr.knist.network.process.compute.function.relu.ReLUD3
import com.wsr.knist.network.process.compute.function.relu.SwishD1
import com.wsr.knist.network.process.compute.function.relu.SwishD2
import com.wsr.knist.network.process.compute.function.relu.SwishD3
import com.wsr.knist.network.process.compute.function.sigmoid.SigmoidD1
import com.wsr.knist.network.process.compute.function.sigmoid.SigmoidD2
import com.wsr.knist.network.process.compute.function.sigmoid.SigmoidD3
import com.wsr.knist.network.process.compute.function.softmax.SoftmaxD1
import com.wsr.knist.network.process.compute.function.softmax.SoftmaxD2
import com.wsr.knist.network.process.compute.function.softmax.SoftmaxD3
import com.wsr.knist.network.process.compute.norm.layer.d1.LayerNormD1
import com.wsr.knist.network.process.compute.norm.layer.d2.LayerNormAxisD2
import com.wsr.knist.network.process.compute.norm.layer.d2.LayerNormD2
import com.wsr.knist.network.process.compute.norm.layer.d3.LayerNormAxisD3
import com.wsr.knist.network.process.compute.norm.layer.d3.LayerNormD3
import com.wsr.knist.network.process.compute.norm.minmax.MinMaxNormD1
import com.wsr.knist.network.process.compute.norm.minmax.MinMaxNormD2
import com.wsr.knist.network.process.compute.norm.minmax.MinMaxNormD3
import com.wsr.knist.network.process.compute.norm.rms.d1.RmsNormD1
import com.wsr.knist.network.process.compute.norm.rms.d2.RmsNormAxisD2
import com.wsr.knist.network.process.compute.norm.rms.d2.RmsNormD2
import com.wsr.knist.network.process.compute.norm.rms.d3.RmsNormAxisD3
import com.wsr.knist.network.process.compute.norm.rms.d3.RmsNormD3
import com.wsr.knist.network.process.compute.pool.MaxPoolD2
import com.wsr.knist.network.process.compute.pool.MaxPoolD3
import com.wsr.knist.network.process.compute.position.PositionEmbeddingD2
import com.wsr.knist.network.process.compute.position.PositionEncodeD2
import com.wsr.knist.network.process.compute.position.RoPED2
import com.wsr.knist.network.process.compute.scale.d1.ScaleD1
import com.wsr.knist.network.process.compute.scale.d2.ScaleAxisD2
import com.wsr.knist.network.process.compute.scale.d2.ScaleD2
import com.wsr.knist.network.process.compute.scale.d3.ScaleAxisD3
import com.wsr.knist.network.process.compute.scale.d3.ScaleD3
import com.wsr.knist.network.process.compute.skip.SkipD1
import com.wsr.knist.network.process.compute.skip.SkipD2
import com.wsr.knist.network.process.compute.skip.SkipD3
import com.wsr.knist.network.process.reshape.gad.GlobalAverageD2ToD1
import com.wsr.knist.network.process.reshape.gad.GlobalAverageD3ToD1
import com.wsr.knist.network.process.reshape.gad.GlobalAverageD3ToD2
import com.wsr.knist.network.process.reshape.reshape.ReshapeD1ToD2
import com.wsr.knist.network.process.reshape.reshape.ReshapeD1ToD3
import com.wsr.knist.network.process.reshape.reshape.ReshapeD2ToD1
import com.wsr.knist.network.process.reshape.reshape.ReshapeD2ToD3
import com.wsr.knist.network.process.reshape.reshape.ReshapeD3ToD1
import com.wsr.knist.network.process.reshape.reshape.ReshapeD3ToD2
import com.wsr.knist.network.process.reshape.token.TokenEmbeddingD1ToD2
import kotlin.jvm.JvmName
import kotlin.reflect.KClass
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

class GraphNetworkSerializer<T : GraphNetwork<T>>(
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

@Suppress("UNCHECKED_CAST")
class NetworkSerializer<I, O> :
    KSerializer<Network<I, O>> by GraphNetworkSerializer(
        factory = { sources, graph, sinks, optimizer, initializer ->
            check(sources.size == 1) { "invalid Network format. sources.size=${sources.size}, expected 1" }
            check(sinks.size == 1) { "invalid Network format. sinks.size=${sinks.size}, expected 1" }
            Network(
                source = sources[0] as Graph.Source<I>,
                graph = graph,
                sink = sinks[0] as Graph.Sink<O>,
                optimizer = optimizer,
                initializer = initializer,
            )
        },
    ) {
    companion object {
        fun <I, O> encodeToString(value: Network<I, O>) = json.encodeToString(
            serializer = NetworkSerializer(),
            value = value,
        )

        fun <I, O> encodeToBufferedSink(value: Network<I, O>, sink: BufferedSink) = json.encodeToBufferedSink(
            serializer = NetworkSerializer(),
            value = value,
            sink = sink,
        )

        fun <I, O> decodeFromString(value: String) = json.decodeFromString<Network<I, O>>(
            deserializer = NetworkSerializer(),
            string = value,
        )

        fun <I, O> decodeFromBufferedSource(source: BufferedSource) = json.decodeFromBufferedSource<Network<I, O>>(
            deserializer = NetworkSerializer(),
            source = source,
        )

        fun <I, O> encodeToCbor(value: Network<I, O>): ByteArray = cbor.encodeToByteArray(NetworkSerializer(), value)

        fun <I, O> encodeToCborSink(value: Network<I, O>, sink: BufferedSink) {
            sink.write(cbor.encodeToByteArray(NetworkSerializer<I, O>(), value))
        }

        fun <I, O> decodeFromCbor(bytes: ByteArray): Network<I, O> =
            cbor.decodeFromByteArray(NetworkSerializer(), bytes)

        fun <I, O> decodeFromCborSource(source: BufferedSource): Network<I, O> =
            cbor.decodeFromByteArray(NetworkSerializer(), source.readByteArray())

        val modules = mutableListOf(buildInSerializersModule)

        @JvmName("registerProcess")
        inline fun <reified T : Process> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(Process::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerOptimizer")
        inline fun <reified T : Optimizer> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(Optimizer::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerWeightInitializer")
        inline fun <reified T : WeightInitializer> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(WeightInitializer::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerOptimizerD1")
        inline fun <reified T : Optimizer.D1> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(Optimizer.D1::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerOptimizerD2")
        inline fun <reified T : Optimizer.D2> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(Optimizer.D2::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerOptimizerD3")
        inline fun <reified T : Optimizer.D3> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(Optimizer.D3::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerAttentionBiasD2")
        inline fun <reified T : AttentionBiasD2> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(AttentionBiasD2::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        @JvmName("registerConverter")
        inline fun <reified T : Converter<*>> register(clazz: KClass<T>) {
            val module = SerializersModule {
                polymorphic(Converter::class) {
                    subclass(clazz)
                }
            }
            modules.add(module)
        }

        private val json
            get() = Json {
                serializersModule = modules.reduce { acc, module -> acc + module }
            }

        private val cbor
            get() = Cbor {
                serializersModule = modules.reduce { acc, module -> acc + module }
            }
    }
}

private val buildInSerializersModule = SerializersModule {
    polymorphic(Process::class) {
        // Affine
        subclass(AffineD1::class)
        subclass(AffineD2::class)

        // Attention
        subclass(AttentionD2::class)

        // Bias
        subclass(BiasD1::class)
        subclass(BiasD2::class)
        subclass(BiasAxisD2::class)
        subclass(BiasD3::class)
        subclass(BiasAxisD3::class)

        // Conv
        subclass(ConvD1::class)
        subclass(ConvD2::class)

        // Debug
        subclass(DebugD1::class)
        subclass(DebugD2::class)
        subclass(DebugD3::class)

        // Dropout
        subclass(DropoutD1::class)
        subclass(DropoutD2::class)
        subclass(DropoutD3::class)

        // Function
        subclass(com.wsr.knist.network.process.compute.function.linear.LinearD1::class)
        subclass(com.wsr.knist.network.process.compute.function.linear.LinearD2::class)
        subclass(com.wsr.knist.network.process.compute.function.linear.LinearD3::class)

        subclass(ReLUD1::class)
        subclass(ReLUD2::class)
        subclass(ReLUD3::class)
        subclass(LeakyReLUD1::class)
        subclass(LeakyReLUD2::class)
        subclass(LeakyReLUD3::class)
        subclass(SwishD1::class)
        subclass(SwishD2::class)
        subclass(SwishD3::class)

        subclass(SigmoidD1::class)
        subclass(SigmoidD2::class)
        subclass(SigmoidD3::class)

        subclass(SoftmaxD1::class)
        subclass(SoftmaxD2::class)
        subclass(SoftmaxD3::class)

        // Norm
        subclass(LayerNormD1::class)
        subclass(LayerNormD2::class)
        subclass(LayerNormAxisD2::class)
        subclass(LayerNormD3::class)
        subclass(LayerNormAxisD3::class)

        subclass(RmsNormD1::class)
        subclass(RmsNormD2::class)
        subclass(RmsNormAxisD2::class)
        subclass(RmsNormD3::class)
        subclass(RmsNormAxisD3::class)

        subclass(MinMaxNormD1::class)
        subclass(MinMaxNormD2::class)
        subclass(MinMaxNormD3::class)

        // Pool
        subclass(MaxPoolD2::class)
        subclass(MaxPoolD3::class)

        // Position
        subclass(PositionEncodeD2::class)
        subclass(PositionEmbeddingD2::class)
        subclass(RoPED2::class)

        // Scale
        subclass(ScaleD1::class)
        subclass(ScaleD2::class)
        subclass(ScaleAxisD2::class)
        subclass(ScaleD3::class)
        subclass(ScaleAxisD3::class)

        // Skip
        subclass(SkipD1::class)
        subclass(SkipD2::class)
        subclass(SkipD3::class)

        // Global Average
        subclass(GlobalAverageD2ToD1::class)
        subclass(GlobalAverageD3ToD1::class)
        subclass(GlobalAverageD3ToD2::class)

        // Reshape
        subclass(ReshapeD1ToD2::class)
        subclass(ReshapeD1ToD3::class)
        subclass(ReshapeD2ToD1::class)
        subclass(ReshapeD2ToD3::class)
        subclass(ReshapeD3ToD1::class)
        subclass(ReshapeD3ToD2::class)

        // Token
        subclass(TokenEmbeddingD1ToD2::class)
    }

    polymorphic(Join::class) {
        subclass(AddD1::class)
        subclass(AddD2::class)
        subclass(AddD3::class)
    }

    polymorphic(Output::class) {
        subclass(MeanSquareD1::class)
        subclass(MeanSquareD2::class)

        subclass(SigmoidWithLossD1::class)
        subclass(SigmoidWithLossD2::class)

        subclass(SoftmaxWithLossD1::class)
        subclass(SoftmaxWithLossD2::class)
    }

    polymorphic(Optimizer::class) {
        subclass(Freeze::class)
        subclass(Sgd::class)
        subclass(Momentum::class)
        subclass(RmsProp::class)
        subclass(Adam::class)
        subclass(AdamW::class)
    }

    polymorphic(Optimizer.D1::class) {
        subclass(FreezeD1::class)
        subclass(SgdD1::class)
        subclass(MomentumD1::class)
        subclass(RmsPropD1::class)
        subclass(AdamD1::class)
        subclass(AdamWD1::class)
    }

    polymorphic(Optimizer.D2::class) {
        subclass(FreezeD2::class)
        subclass(SgdD2::class)
        subclass(MomentumD2::class)
        subclass(RmsPropD2::class)
        subclass(AdamD2::class)
        subclass(AdamWD2::class)
    }

    polymorphic(Optimizer.D3::class) {
        subclass(FreezeD3::class)
        subclass(SgdD3::class)
        subclass(MomentumD3::class)
        subclass(RmsPropD3::class)
        subclass(AdamD3::class)
        subclass(AdamWD3::class)
    }

    polymorphic(Optimizer.D4::class) {
        subclass(FreezeD4::class)
        subclass(SgdD4::class)
        subclass(MomentumD4::class)
        subclass(RmsPropD4::class)
        subclass(AdamD4::class)
        subclass(AdamWD4::class)
    }

    polymorphic(Scheduler::class) {
        subclass(Scheduler.Fix::class)
        subclass(Scheduler.Step::class)
        subclass(Scheduler.MultiStep::class)
        subclass(Scheduler.CosineAnnealing::class)
    }

    polymorphic(WeightInitializer::class) {
        subclass(He::class)
        subclass(Xavier::class)
        subclass(Random::class)
        subclass(Fixed::class)
    }

    polymorphic(Converter::class) {
        // Raw
        subclass(RawD1::class)
        subclass(RawD2::class)
        subclass(RawD3::class)
    }

    polymorphic(AttentionBiasD2::class) {
        subclass(AttentionBiasD2.Causal::class)
        subclass(AttentionBiasD2.Mask::class)
        subclass(AttentionBiasD2.ALiBi::class)
    }

    polymorphic(Graph.Node::class) {
        subclass(Graph.Node.Attach::class)
        subclass(Graph.Node.Connect::class)
        subclass(Graph.Node.Observe::class)
    }
}
