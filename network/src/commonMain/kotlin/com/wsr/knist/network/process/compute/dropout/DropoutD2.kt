package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.times.times
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.nextFloat
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.random.Random
import kotlinx.serialization.Serializable

@Serializable
class DropoutD2 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    private val ratio: Float,
    private val seed: Int? = null,
) : Compute.D2() {
    private val random by lazy { seed?.let { Random(it) } ?: Random }
    private val q = 1 / ratio

    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val mask = IOType.d2(outputX, outputY) { _, _ ->
            if (random.nextFloat(0f, 1f) <= ratio) q else 0f
        }
        val output = input * mask
        val delta = calcDelta(output)
        return delta * mask
    }
}

fun <T> NetworkBuilder.D2<T>.dropout(ratio: Float, seed: Int? = null) = addProcess(
    process =
        DropoutD2(
            outputX = inputX,
            outputY = inputY,
            ratio = ratio,
            seed = seed,
        ),
)
