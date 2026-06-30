package com.wsr.knist.network.process.compute.attention.bias

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.compare.eq
import com.wsr.knist.batch.elementwise.compare.where.where
import com.wsr.knist.batch.elementwise.operation.plus.plus
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.network.process.Context
import kotlinx.serialization.Serializable

@Serializable
sealed interface AttentionBiasD2 {
    fun IOScope.forward(scaled: Batch<IOType.D3>, context: Context): Batch<IOType.D3>
    fun IOScope.backward(delta: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = delta

    @Serializable
    data class Casual(val inputI: Int) : AttentionBiasD2 {
        private val mask by lazy {
            IOType.d2(inputI, inputI) { x, y -> if (x < y) -1e9f else 0f }
        }

        override fun IOScope.forward(scaled: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = scaled + mask
    }

    @Serializable
    data class Mask(val value: Float) : AttentionBiasD2 {
        override fun IOScope.forward(scaled: Batch<IOType.D3>, context: Context): Batch<IOType.D3> {
            @Suppress("UNCHECKED_CAST")
            val input = context.input as Batch<IOType.D1>
            val mask = input.where(onTrue = -1e9f, onFalse = 0f) { it eq value }
            return scaled.plus(other = mask, axis = 2)
        }
    }
}

data class AttentionBiasD2Builder(
    val inputI: Int,
    val inputJ: Int,
    val numOfHeads: Int,
    val biases: List<AttentionBiasD2> = emptyList(),
)
