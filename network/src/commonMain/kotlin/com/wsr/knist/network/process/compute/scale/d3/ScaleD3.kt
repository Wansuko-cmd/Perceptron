package com.wsr.knist.network.process.compute.scale.d3

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class ScaleD3 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
    private val optimizer: Optimizer.D3,
    private var weight: IOType.D3.Global,
) : Compute.D3() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input * weight

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val output = input * weight
        val delta = calcDelta(output)

        val dx = delta * weight
        weight = optimizer.adapt(
            weight = weight,
            dw = input * delta,
        ).toGlobal()

        return dx
    }
}

fun <T> NetworkBuilder.D3<T>.scale(
    axis: Int? = null,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(1f),
): NetworkBuilder.D3<T> {
    val process = when (axis) {
        null -> ScaleD3(
            outputX = inputX,
            outputY = inputY,
            outputZ = inputZ,
            optimizer = optimizer.d3(
                inputX,
                inputY,
                inputZ,
            ),
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
            ScaleAxisD3(
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
