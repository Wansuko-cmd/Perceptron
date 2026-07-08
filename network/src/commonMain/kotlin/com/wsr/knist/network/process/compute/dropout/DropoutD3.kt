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
class DropoutD3 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val outputK: Int,
    private val ratio: Float,
    private val seed: Int? = null,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    private val random by lazy { seed?.let { Random(it) } ?: Random }
    private val q = 1 / ratio

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val uniform = Batch.random(
            size = input.size,
            i = outputI,
            j = outputJ,
            k = outputK,
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

fun <T> NetworkBuilder.D3<T>.dropout(ratio: Float, seed: Int? = null, id: String = Uuid.random().toString()) =
    addProcess(
        process =
            DropoutD3(
                outputI = inputI,
                outputJ = inputJ,
                outputK = inputK,
                ratio = ratio,
                seed = seed,
                id = id,
            ),
    )
