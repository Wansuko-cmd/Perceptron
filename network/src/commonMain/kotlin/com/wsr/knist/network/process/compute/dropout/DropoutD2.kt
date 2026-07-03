package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.random.Random
import kotlinx.serialization.Serializable

@Serializable
class DropoutD2 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    private val ratio: Float,
    private val seed: Int? = null,
) : Compute.D2() {
    private val random by lazy { seed?.let { Random(it) } ?: Random }
    private val q = 1 / ratio

    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val uniform = Batch.random(
            size = input.size,
            i = outputI,
            j = outputJ,
            from = 0f,
            until = 1f,
            random = random,
        )
        val mask = uniform.where(condition = uniform lt ratio, onTrue = q, onFalse = 0f)
        val output = input * mask
        val delta = calcDelta(output)
        return delta * mask
    }
}

fun <T> NetworkBuilder.D2<T>.dropout(ratio: Float, seed: Int? = null) = addProcess(
    process =
        DropoutD2(
            outputI = inputI,
            outputJ = inputJ,
            ratio = ratio,
            seed = seed,
        ),
)
