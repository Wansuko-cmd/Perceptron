package com.wsr.knist.network.initializer

import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.network.nextFloat
import kotlin.math.sqrt
import kotlin.random.Random
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

@Serializable
class He(private val seed: Int? = null) : WeightInitializer {
    @Transient
    private val random = seed?.let { Random(it) } ?: Random
    override fun d1(input: List<Int>, output: List<Int>, size: Int): IOType.D1.Global {
        val weight = calcWeight(
            fanIn = input.reduce { acc, i -> acc * i },
            size = size,
        )
        return IOType.d1(value = weight)
    }

    override fun d2(input: List<Int>, output: List<Int>, i: Int, j: Int): IOType.D2.Global {
        val weight = calcWeight(
            fanIn = input.reduce { acc, i -> acc * i },
            size = i * j,
        )
        return IOType.d2(shape = listOf(i, j), value = weight)
    }

    override fun d3(input: List<Int>, output: List<Int>, i: Int, j: Int, k: Int): IOType.D3.Global {
        val weight = calcWeight(
            fanIn = input.reduce { acc, i -> acc * i },
            size = i * j * k,
        )
        return IOType.d3(shape = listOf(i, j, k), value = weight)
    }

    override fun d4(input: List<Int>, output: List<Int>, i: Int, j: Int, k: Int, l: Int): IOType.D4.Global {
        val weight = calcWeight(
            fanIn = input.reduce { acc, i -> acc * i },
            size = i * j * k * l,
        )
        return IOType.d4(shape = listOf(i, j, k, l), value = weight)
    }

    private fun calcWeight(fanIn: Int, size: Int): FloatArray {
        val limit = sqrt(6f / fanIn)
        return FloatArray(size) { random.nextFloat(-limit, limit) }
    }
}
