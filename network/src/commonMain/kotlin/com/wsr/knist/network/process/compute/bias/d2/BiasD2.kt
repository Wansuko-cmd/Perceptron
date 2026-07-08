package com.wsr.knist.network.process.compute.bias.d2

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class BiasD2(
    override val outputI: Int,
    override val outputJ: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2.Global,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input + weight

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input + weight
        val delta = calcDelta(output)
        weight = optimizer.adapt(weight = weight, dw = delta).toGlobal()
        return delta
    }
}

fun <T> NetworkBuilder.D2<T>.bias(
    axis: Int? = null,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(0f),
    id: String = Uuid.random().toString(),
): NetworkBuilder.D2<T> {
    val process = when (axis) {
        null -> BiasD2(
            outputI = inputI,
            outputJ = inputJ,
            optimizer = optimizer.d2(inputI, inputJ),
            weight = initializer.d2(
                input = listOf(inputI, inputJ),
                output = listOf(inputI, inputJ),
                i = inputI,
                j = inputJ,
            ),
            id = id,
        )

        0, 1 -> {
            val inputT = if (axis == 0) inputI else inputJ
            BiasAxisD2(
                outputI = inputI,
                outputJ = inputJ,
                axis = axis,
                optimizer = optimizer.d1(inputT),
                weight = initializer.d1(
                    input = listOf(inputT),
                    output = listOf(inputT),
                    size = inputT,
                ),
                id = id,
            )
        }

        else -> throw IllegalStateException(
            """
            invalid parameter.
            axis: $axis
            """.trimIndent(),
        )
    }
    return addProcess(process = process)
}
