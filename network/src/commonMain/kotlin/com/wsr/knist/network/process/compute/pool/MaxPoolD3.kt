package com.wsr.knist.network.process.compute.pool

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.broadcastToD4
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class MaxPoolD3 internal constructor(
    val poolSize: Int,
    val channel: Int,
    val inputX: Int,
    val inputY: Int,
    val padding: Int,
) : Compute.D3() {
    override val outputI: Int = channel
    override val outputJ: Int = (inputX + 2 * padding - poolSize) / poolSize + 1
    override val outputK: Int = (inputY + 2 * padding - poolSize) / poolSize + 1

    init {
        check(
            (inputX + 2 * padding - poolSize) % poolSize == 0 &&
                (inputY + 2 * padding - poolSize) % poolSize == 0,
        ) {
            """
            invalid parameter.
            inputX: $inputX
            inputY: $inputY
            poolSize: $poolSize
            padding: $padding
            outputY: ${(inputX + 2 * padding - poolSize) / poolSize.toFloat() + 1}
            outputZ: ${(inputY + 2 * padding - poolSize) / poolSize.toFloat() + 1}
            """.trimIndent()
        }
    }

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input.unfold(
        windowSize = poolSize,
        stride = poolSize,
        padding = padding,
    )
        .max(axis = 3)

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val unfold = input.unfold(windowSize = poolSize, stride = poolSize, padding = padding)
        val output = unfold.max(axis = 3)
        val delta = calcDelta(output)
        val windowSize = poolSize * poolSize
        return where(
            condition = unfold eq output.broadcastToD4(axis = 3, size = windowSize),
            onTrue = delta.broadcastToD4(axis = 3, size = windowSize),
            onFalse = 0f,
        ).fold(stride = poolSize, padding = padding)
    }
}

fun <T> NetworkBuilder.D3<T>.maxPool(size: Int, padding: Int = 0) = addProcess(
    process = MaxPoolD3(
        poolSize = size,
        channel = inputX,
        inputX = inputY,
        inputY = inputZ,
        padding = padding,
    ),
)
