package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.random.Random
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class DropoutD1 internal constructor(
    override val outputI: Int,
    private val ratio: Float,
    private val seed: Int? = null,
    override val id: String = Uuid.random().toString(),
) : Compute.D1() {
    private val random by lazy { seed?.let { Random(it) } ?: Random }
    private val q = 1 / ratio

    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> = input

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val uniform = Batch.random(
            size = input.size,
            i = outputI,
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

fun <T> NetworkBuilder.D1<T>.dropout(ratio: Float, seed: Int? = null, id: String = Uuid.random().toString()) =
    addProcess(
        process =
            DropoutD1(
                outputI = inputI,
                ratio = ratio,
                seed = seed,
                id = id,
            ),
    )
