package com.wsr.knist.network.process.compute.pool

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class MaxPoolD3 internal constructor(
    val poolSize: Int,
    val channel: Int,
    val inputI: Int,
    val inputJ: Int,
    val padding: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    override val outputI: Int = channel
    override val outputJ: Int = (inputI + 2 * padding - poolSize) / poolSize + 1
    override val outputK: Int = (inputJ + 2 * padding - poolSize) / poolSize + 1

    init {
        check(
            (inputI + 2 * padding - poolSize) % poolSize == 0 &&
                (inputJ + 2 * padding - poolSize) % poolSize == 0,
        ) {
            """
            invalid parameter.
            inputI: $inputI
            inputJ: $inputJ
            poolSize: $poolSize
            padding: $padding
            outputJ: ${(inputI + 2 * padding - poolSize) / poolSize.toFloat() + 1}
            outputK: ${(inputJ + 2 * padding - poolSize) / poolSize.toFloat() + 1}
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

fun <T> NetworkBuilder.D3<T>.maxPool(size: Int, padding: Int = 0, id: String = Uuid.random().toString()) = addProcess(
    process = MaxPoolD3(
        poolSize = size,
        channel = inputI,
        inputI = inputJ,
        inputJ = inputK,
        padding = padding,
        id = id,
    ),
)
