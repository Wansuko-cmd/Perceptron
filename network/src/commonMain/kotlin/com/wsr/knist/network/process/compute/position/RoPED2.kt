package com.wsr.knist.network.process.compute.position

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.minus.minus
import com.wsr.knist.batch.elementwise.operation.plus.plus
import com.wsr.knist.batch.elementwise.operation.times.times
import com.wsr.knist.batch.shape.interleave
import com.wsr.knist.batch.shape.slice
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlinx.serialization.Serializable

@Serializable
class RoPED2 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    private val waveLength: Float,
) : Compute.D2() {
    private val theta by lazy {
        IOType.d1(outputY / 2) { i -> 1f / waveLength.pow(2f * i / outputY) }
    }

    private val cosCache by lazy {
        IOType.d2(outputX, outputY / 2) { x, y -> cos(x * theta[y].get()) }
    }

    private val sinCache by lazy {
        IOType.d2(outputX, outputY / 2) { x, y -> sin(x * theta[y].get()) }
    }

    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input.applyRoPE()

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input.applyRoPE()
        val delta = calcDelta(output)
        return delta.applyRoPE()
    }

    private fun Batch<IOType.D2>.applyRoPE(): Batch<IOType.D2> {
        val even = this.slice(0 until shape[1] step 2, axis = 1)
        val odd = this.slice(1 until shape[1] step 2, axis = 1)

        val resEven = even * cosCache - odd * sinCache
        val resOdd = even * sinCache + odd * cosCache

        return resEven.interleave(resOdd, axis = 1)
    }
}

fun <T> NetworkBuilder.D2<T>.roPE(waveLength: Float = 10000f) = addProcess(
    process = RoPED2(
        outputX = inputX,
        outputY = inputY,
        waveLength = waveLength,
    ),
)
