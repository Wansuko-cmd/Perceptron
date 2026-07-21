package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmInline

@JvmInline
value class GraphEnv internal constructor(private val values: MutableMap<GraphId, Batch<IOType>> = mutableMapOf()) {
    @Suppress("UNCHECKED_CAST")
    operator fun <T : IOType> get(id: GraphId): Batch<T> = values[id]?.let { it as Batch<T> }
        ?: error("GraphEnv: value not found for id=$id")

    operator fun <T : IOType> set(id: GraphId, value: Batch<T>) {
        @Suppress("UNCHECKED_CAST")
        values[id] = value as Batch<IOType>
    }
}