package com.wsr.knist.network

import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.process.Reshape

object GraphBuilder {
    sealed interface Node {
        val from: GraphId
        val nodes: List<Graph.Node>
        val optimizer: Optimizer
        val initializer: WeightInitializer

        data class D1(
            val inputI: Int,
            override val from: GraphId,
            override val nodes: List<Graph.Node>,
            override val optimizer: Optimizer,
            override val initializer: WeightInitializer,
        ) : Node

        data class D2(
            val inputI: Int,
            val inputJ: Int,
            override val from: GraphId,
            override val nodes: List<Graph.Node>,
            override val optimizer: Optimizer,
            override val initializer: WeightInitializer,
        ) : Node

        data class D3(
            val inputI: Int,
            val inputJ: Int,
            val inputK: Int,
            override val from: GraphId,
            override val nodes: List<Graph.Node>,
            override val optimizer: Optimizer,
            override val initializer: WeightInitializer,
        ) : Node
    }

    data class Result<O>(val nodes: List<Graph.Node>, val sink: Graph.Sink<O>)
}

object GraphScope {
    fun GraphBuilder.Node.D1.addCompute(compute: Compute.D1): GraphBuilder.Node.D1 {
        val node = Graph.Node.Attach(from = from, process = compute)
        return GraphBuilder.Node.D1(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = compute.outputI,
        )
    }

    fun GraphBuilder.Node.D2.addCompute(compute: Compute.D2): GraphBuilder.Node.D2 {
        val node = Graph.Node.Attach(from = from, process = compute)
        return GraphBuilder.Node.D2(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = compute.outputI,
            inputJ = compute.outputJ,
        )
    }

    fun GraphBuilder.Node.D3.addCompute(compute: Compute.D3): GraphBuilder.Node.D3 {
        val node = Graph.Node.Attach(from = from, process = compute)
        return GraphBuilder.Node.D3(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = compute.outputI,
            inputJ = compute.outputJ,
            inputK = compute.outputK,
        )
    }

    fun GraphBuilder.Node.D1.addReshape(reshape: Reshape.D1ToD2): GraphBuilder.Node.D2 {
        val node = Graph.Node.Attach(from = from, process = reshape)
        return GraphBuilder.Node.D2(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
        )
    }

    fun GraphBuilder.Node.D1.addReshape(reshape: Reshape.D1ToD3): GraphBuilder.Node.D3 {
        val node = Graph.Node.Attach(from = from, process = reshape)
        return GraphBuilder.Node.D3(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
            inputK = reshape.outputK,
        )
    }

    fun GraphBuilder.Node.D2.addReshape(reshape: Reshape.D2ToD1): GraphBuilder.Node.D1 {
        val node = Graph.Node.Attach(from = from, process = reshape)
        return GraphBuilder.Node.D1(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = reshape.outputI,
        )
    }

    fun GraphBuilder.Node.D2.addReshape(reshape: Reshape.D2ToD3): GraphBuilder.Node.D3 {
        val node = Graph.Node.Attach(from = from, process = reshape)
        return GraphBuilder.Node.D3(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
            inputK = reshape.outputK,
        )
    }

    fun GraphBuilder.Node.D3.addReshape(reshape: Reshape.D3ToD1): GraphBuilder.Node.D1 {
        val node = Graph.Node.Attach(from = from, process = reshape)
        return GraphBuilder.Node.D1(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = reshape.outputI,
        )
    }

    fun GraphBuilder.Node.D3.addReshape(reshape: Reshape.D3ToD2): GraphBuilder.Node.D2 {
        val node = Graph.Node.Attach(from = from, process = reshape)
        return GraphBuilder.Node.D2(
            from = node.id,
            nodes = nodes + node,
            optimizer = optimizer,
            initializer = initializer,
            inputI = reshape.outputI,
            inputJ = reshape.outputJ,
        )
    }

    fun <O> GraphBuilder.Node.D1.addOutput(output: Output.D1, converter: Converter<O>): GraphBuilder.Result<O> {
        val sink = Graph.Sink(from = from, output = output, converter = converter)
        return GraphBuilder.Result(nodes = nodes, sink = sink)
    }

    fun <O> GraphBuilder.Node.D2.addOutput(output: Output.D2, converter: Converter<O>): GraphBuilder.Result<O> {
        val sink = Graph.Sink(from = from, output = output, converter = converter)
        return GraphBuilder.Result(nodes = nodes, sink = sink)
    }
}

fun <I, O> NetworkV2.Companion.create(
    converter: Converter.D1<I>,
    optimizer: Optimizer,
    initializer: WeightInitializer,
    block: GraphScope.(GraphBuilder.Node.D1) -> GraphBuilder.Result<O>,
): NetworkV2<I, O> {
    val source = Graph.Source(converter = converter)
    val builder = GraphBuilder.Node.D1(
        from = source.id,
        nodes = listOf(),
        optimizer = optimizer,
        initializer = initializer,
        inputI = converter.outputI,
    )
    val result = GraphScope.block(builder)
    return NetworkV2(
        source = source,
        graph = result.nodes,
        sink = result.sink,
        optimizer = optimizer,
        initializer = initializer,
    )
}

fun <I, O> NetworkV2.Companion.create(
    converter: Converter.D2<I>,
    optimizer: Optimizer,
    initializer: WeightInitializer,
    block: GraphScope.(GraphBuilder.Node.D2) -> GraphBuilder.Result<O>,
): NetworkV2<I, O> {
    val source = Graph.Source(converter = converter)
    val builder = GraphBuilder.Node.D2(
        from = source.id,
        nodes = listOf(),
        optimizer = optimizer,
        initializer = initializer,
        inputI = converter.outputI,
        inputJ = converter.outputJ,
    )
    val result = GraphScope.block(builder)
    return NetworkV2(
        source = source,
        graph = result.nodes,
        sink = result.sink,
        optimizer = optimizer,
        initializer = initializer,
    )
}
fun <I, O> NetworkV2.Companion.create(
    converter: Converter.D3<I>,
    optimizer: Optimizer,
    initializer: WeightInitializer,
    block: GraphScope.(GraphBuilder.Node.D3) -> GraphBuilder.Result<O>,
): NetworkV2<I, O> {
    val source = Graph.Source(converter = converter)
    val builder = GraphBuilder.Node.D3(
        from = source.id,
        nodes = listOf(),
        optimizer = optimizer,
        initializer = initializer,
        inputI = converter.outputI,
        inputJ = converter.outputJ,
        inputK = converter.outputK,
    )
    val result = GraphScope.block(builder)
    return NetworkV2(
        source = source,
        graph = result.nodes,
        sink = result.sink,
        optimizer = optimizer,
        initializer = initializer,
    )
}
