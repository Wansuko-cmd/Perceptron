package com.wsr.knist.network.process.compute.norm.rms.d2

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.process.Context
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class RmsNormD2 internal constructor(
    override val inputI: Int,
    override val inputJ: Int,
    private val e: Float,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override val outputI: Int get() = inputI
    override val outputJ: Int get() = inputJ
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val deviation = input.pow(n = 2).average().sqrt(e = e)
        return input / deviation
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val variance = input.pow(2).average()
        val deviation = variance.sqrt(e = e)
        val output = input / deviation

        val delta = calcDelta(output)

        val dx1 = delta / deviation
        val dx2 = run {
            val m = (delta * input).average() / deviation.pow(3)
            m * input
        }
        return dx1 - dx2
    }
}

fun <T> NetworkBuilder.D2<T>.rmsNorm(
    axis: Int? = null,
    e: Float = 1e-6f,
    id: String = Uuid.random().toString(),
): NetworkBuilder.D2<T> {
    val process = when (axis) {
        null -> RmsNormD2(
            inputI = inputI,
            inputJ = inputJ,
            e = e,
            id = id,
        )

        0, 1 -> RmsNormAxisD2(
            inputI = inputI,
            inputJ = inputJ,
            axis = axis,
            e = e,
            id = id,
        )

        else -> throw IllegalStateException(
            """
            invalid parameter.
            axis: $axis
            """.trimIndent(),
        )
    }
    return addCompute(compute = process)
}
