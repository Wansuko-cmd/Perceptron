package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.plus.plus
import com.wsr.knist.core.IOScope
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

    context(scope: IOScope)
    @Suppress("UNCHECKED_CAST")
    fun <T : IOType> plus(id: GraphId, value: Batch<T>) {
        val current = values[id]
        values[id] = if (current == null) {
            value as Batch<IOType>
        } else {
            accumulate(current, value as Batch<IOType>)
        }
    }

    fun reset() {
        values.clear()
    }
}

context(scope: IOScope)
@Suppress("UNCHECKED_CAST")
private fun accumulate(a: Batch<IOType>, b: Batch<IOType>): Batch<IOType> {
    check(a.shape == b.shape) { "GraphEnv: shape mismatch on accumulate. a=$a.shape, b=$b.shape" }
    return when (a.shape.size) {
        1 -> ((a as Batch<IOType.D1>) + (b as Batch<IOType.D1>)) as Batch<IOType>
        2 -> ((a as Batch<IOType.D2>) + (b as Batch<IOType.D2>)) as Batch<IOType>
        3 -> ((a as Batch<IOType.D3>) + (b as Batch<IOType.D3>)) as Batch<IOType>
        else -> error("GraphEnv: unsupported rank for accumulate: $a.shape")
    }
}