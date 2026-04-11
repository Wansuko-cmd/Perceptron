package com.wsr.network.process.compute.bias.d3

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
class BiasD3(
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
    private val optimizer: Optimizer.D3,
    private var weight: IOType.D3,
) : Compute.D3() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input + weight

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val output = input + weight
        val delta = calcDelta(output)
        weight = optimizer.adapt(weight = weight, dw = delta)
        return delta
    }
}

fun <T> NetworkBuilder.D3<T>.bias(
    axis: Int? = null,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(0f),
): NetworkBuilder.D3<T> {
    val process = when (axis) {
        null -> BiasD3(
            outputX = inputX,
            outputY = inputY,
            outputZ = inputZ,
            optimizer = optimizer.d3(inputX, inputY, inputZ),
            weight = initializer.d3(
                input = listOf(inputX, inputY, inputZ),
                output = listOf(inputX, inputY, inputZ),
                x = inputX,
                y = inputY,
                z = inputZ,
            ),
        )

        0, 1, 2 -> {
            val inputT = when (axis) {
                0 -> inputX
                1 -> inputY
                else -> inputZ
            }
            BiasAxisD3(
                outputX = inputX,
                outputY = inputY,
                outputZ = inputZ,
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
