package com.wsr.knist.network.process.compute.pool

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.compare.eq
import com.wsr.knist.batch.elementwise.compare.where.where
import com.wsr.knist.batch.reduction.max
import com.wsr.knist.batch.shape.broadcastToD4
import com.wsr.knist.batch.shape.fold.fold
import com.wsr.knist.batch.shape.fold.unfold
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class MaxPoolD3 internal constructor(val poolSize: Int, val channel: Int, val inputX: Int, val inputY: Int) :
    Compute.D3() {
    override val outputX: Int = channel
    override val outputY: Int = inputX / poolSize
    override val outputZ: Int = inputY / poolSize

    init {
        check(inputX % poolSize == 0 && inputY % poolSize == 0) {
            """
            invalid parameter.
            inputX: $inputX
            inputY: $inputY
            poolSize: $poolSize
            outputX: ${inputX / poolSize.toFloat()}
            outputY: ${inputY / poolSize.toFloat()}
            """.trimIndent()
        }
    }

    override fun expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input.unfold(
        windowSize = poolSize,
        stride = poolSize,
        padding = 0,
    )
        .max(axis = 3)

    override fun train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val unfold = input.unfold(windowSize = poolSize, stride = poolSize, padding = 0)
        val output = unfold.max(axis = 3)
        val delta = calcDelta(output)
        val windowSize = poolSize * poolSize
        return where(
            condition = unfold eq output.broadcastToD4(axis = 3, size = windowSize),
            onTrue = delta.broadcastToD4(axis = 3, size = windowSize),
            onFalse = 0f,
        ).fold(stride = poolSize, padding = 0)
    }
}

fun <T> NetworkBuilder.D3<T>.maxPool(size: Int) = addProcess(
    process = MaxPoolD3(
        poolSize = size,
        channel = inputX,
        inputX = inputY,
        inputY = inputZ,
    ),
)
