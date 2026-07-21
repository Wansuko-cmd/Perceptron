package com.wsr.knist.network

import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Process
import kotlin.jvm.JvmInline
import kotlin.uuid.Uuid

object Graph {
    data class Source<T>(val id: GraphId = GraphId(), val converter: Converter<T>)

    data class Sink<T>(val id: GraphId = GraphId(), val from: GraphId, val output: Output, val converter: Converter<T>)

    sealed interface Node {
        val id: GraphId

        data class Attach(override val id: GraphId = GraphId(), val from: GraphId, val process: Process) : Node
    }
}

@JvmInline
value class GraphId(val value: String = Uuid.random().toString())
