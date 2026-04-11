package com.wsr.network.process.compute.bias.d2

import com.wsr.batch.Batch
import com.wsr.batch.operation.plus.plus
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.initializer.Fixed
import com.wsr.network.initializer.WeightInitializer
import com.wsr.network.optimizer.Optimizer
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class BiasD2(
    override val outputX: Int,
    override val outputY: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2,
) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input + weight

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input + weight
        val delta = calcDelta(output)
        weight = optimizer.adapt(weight = weight, dw = delta)
        return delta
    }
}

fun <T> NetworkBuilder.D2<T>.bias(
    axis: Int? = null,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(0f),
): NetworkBuilder.D2<T> {
    val process = when (axis) {
        null -> BiasD2(
            outputX = inputX,
            outputY = inputY,
            optimizer = optimizer.d2(inputX, inputY),
            weight = initializer.d2(
                input = listOf(inputX, inputY),
                output = listOf(inputX, inputY),
                x = inputX,
                y = inputY,
            ),
        )

        0, 1 -> {
            val inputT = if (axis == 0) inputX else inputY
            BiasAxisD2(
                outputX = inputX,
                outputY = inputY,
                axis = axis,
                optimizer = optimizer.d1(inputT),
                weight = initializer.d1(
                    input = listOf(inputT),
                    output = listOf(inputT),
                    size = inputT,
                ),
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
