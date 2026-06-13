package com.wsr.knist.network.process.compute.pool

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.compare.eq
import com.wsr.knist.batch.elementwise.compare.where.where
import com.wsr.knist.batch.reduction.max
import com.wsr.knist.batch.shape.broadcastToD3
import com.wsr.knist.batch.shape.fold.fold
import com.wsr.knist.batch.shape.fold.unfold
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class MaxPoolD2 internal constructor(val poolSize: Int, val channel: Int, val inputSize: Int) : Compute.D2() {
    override val outputX: Int = channel
    override val outputY: Int = inputSize / poolSize

    init {
        check(inputSize % poolSize == 0) {
            """
            invalid parameter.
            inputSize: $inputSize
            poolSize: $poolSize
            output: ${inputSize / poolSize.toFloat()}
            """.trimIndent()
        }
    }

    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input.unfold(
        windowSize = poolSize,
        stride = poolSize,
        padding = 0,
    )
        .max(axis = 2)

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val unfold = input.unfold(windowSize = poolSize, stride = poolSize, padding = 0)
        val output = unfold.max(axis = 2)
        val delta = calcDelta(output)
        return where(
            condition = unfold eq output.broadcastToD3(axis = 2, size = poolSize),
            onTrue = delta.broadcastToD3(axis = 2, size = poolSize),
            onFalse = 0f,
        ).fold(stride = poolSize, padding = 0)
    }
}

fun <T> NetworkBuilder.D2<T>.maxPool(size: Int) = addProcess(
    process =
    MaxPoolD2(
        poolSize = size,
        channel = inputX,
        inputSize = inputY,
    ),
)
