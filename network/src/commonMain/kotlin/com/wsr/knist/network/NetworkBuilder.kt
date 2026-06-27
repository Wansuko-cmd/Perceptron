package com.wsr.knist.network

import com.wsr.knist.core.IOType
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.converter.linear.LinearD1
import com.wsr.knist.network.converter.linear.LinearD2
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.compute.Compute
import com.wsr.knist.network.process.reshape.Reshape

sealed interface NetworkBuilder<I, O> {
    val input: Converter
    val layers: List<Process>
    val optimizer: Optimizer
    val initializer: WeightInitializer

    @ConsistentCopyVisibility
    data class D1<I> internal constructor(
        val inputI: Int,
        override val input: Converter,
        override val layers: List<Process>,
        override val optimizer: Optimizer,
        override val initializer: WeightInitializer,
    ) : NetworkBuilder<I, IOType.D1> {
        fun addProcess(process: Compute.D1): D1<I> = copy(
            layers = layers + process,
            inputI = process.outputI,
        )

        fun addOutput(output: Output.D1) = Network<I, IOType.D1>(
            inputConverter = input,
            outputConverter = LinearD1(inputI),
            layers = layers,
            output = output,
        )

        fun <O> addOutput(output: Output.D1, converter: D1<I>.() -> Converter.D1<O>) = Network<I, O>(
            inputConverter = input,
            outputConverter = converter(),
            layers = layers,
            output = output,
        )

        fun addReshape(reshape: Reshape.D1ToD2): D2<I> = D2(
            input = input,
            layers = layers + reshape,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
            optimizer = optimizer,
            initializer = initializer,
        )

        fun addReshape(reshape: Reshape.D1ToD3): D3<I> = D3(
            input = input,
            layers = layers + reshape,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
            inputK = reshape.outputK,
            optimizer = optimizer,
            initializer = initializer,
        )

        fun repeat(times: Int, builder: D1<I>.(index: Int) -> D1<I>): D1<I> =
            (0 until times).fold(this) { acc, i -> acc.builder(i) }
    }

    @ConsistentCopyVisibility
    data class D2<I> internal constructor(
        val inputI: Int,
        val inputJ: Int,
        override val input: Converter,
        override val layers: List<Process>,
        override val optimizer: Optimizer,
        override val initializer: WeightInitializer,
    ) : NetworkBuilder<I, IOType.D2> {
        fun addProcess(process: Compute.D2): D2<I> = copy(
            layers = layers + process,
            inputI = process.outputI,
            inputJ = process.outputJ,
        )

        fun addOutput(output: Output.D2) = Network<I, IOType.D2>(
            inputConverter = input,
            outputConverter = LinearD2(inputI, inputJ),
            layers = layers,
            output = output,
        )

        fun <O> addOutput(output: Output.D2, converter: D2<I>.() -> Converter.D2<O>) = Network<I, O>(
            inputConverter = input,
            outputConverter = converter(),
            layers = layers,
            output = output,
        )

        fun addReshape(reshape: Reshape.D2ToD1): D1<I> = D1(
            input = input,
            layers = layers + reshape,
            inputI = reshape.outputI,
            optimizer = optimizer,
            initializer = initializer,
        )

        fun addReshape(reshape: Reshape.D2ToD3): D3<I> = D3(
            input = input,
            layers = layers + reshape,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
            inputK = reshape.outputK,
            optimizer = optimizer,
            initializer = initializer,
        )

        fun repeat(times: Int, builder: D2<I>.(index: Int) -> D2<I>): D2<I> =
            (0 until times).fold(this) { acc, i -> acc.builder(i) }
    }

    @ConsistentCopyVisibility
    data class D3<I> internal constructor(
        val inputI: Int,
        val inputJ: Int,
        val inputK: Int,
        override val input: Converter,
        override val layers: List<Process>,
        override val optimizer: Optimizer,
        override val initializer: WeightInitializer,
    ) : NetworkBuilder<I, IOType.D3> {
        fun addProcess(process: Compute.D3): D3<I> = copy(
            layers = layers + process,
            inputI = process.outputI,
            inputJ = process.outputJ,
            inputK = process.outputK,
        )

        fun addReshape(reshape: Reshape.D3ToD2): D2<I> = D2(
            input = input,
            layers = layers + reshape,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
            optimizer = optimizer,
            initializer = initializer,
        )

        fun addReshape(reshape: Reshape.D3ToD1): D1<I> = D1(
            input = input,
            layers = layers + reshape,
            inputI = reshape.outputI,
            optimizer = optimizer,
            initializer = initializer,
        )

        fun repeat(times: Int, builder: D3<I>.(index: Int) -> D3<I>): D3<I> =
            (0 until times).fold(this) { acc, i -> this.builder(i) }
    }

    companion object {
        fun <T> inputD1(converter: Converter.D1<T>, optimizer: Optimizer, initializer: WeightInitializer) = D1<T>(
            inputI = converter.outputI,
            optimizer = optimizer,
            initializer = initializer,
            input = converter,
            layers = emptyList(),
        )

        fun <T> inputD2(converter: Converter.D2<T>, optimizer: Optimizer, initializer: WeightInitializer) = D2<T>(
            inputI = converter.outputI,
            inputJ = converter.outputJ,
            optimizer = optimizer,
            initializer = initializer,
            input = converter,
            layers = emptyList(),
        )

        fun <T> inputD3(converter: Converter.D3<T>, optimizer: Optimizer, initializer: WeightInitializer) = D3<T>(
            inputI = converter.outputI,
            inputJ = converter.outputJ,
            inputK = converter.outputK,
            optimizer = optimizer,
            initializer = initializer,
            input = converter,
            layers = emptyList(),
        )
    }
}
