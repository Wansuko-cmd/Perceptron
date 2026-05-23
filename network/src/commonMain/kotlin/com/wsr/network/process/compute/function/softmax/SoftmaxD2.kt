package com.wsr.network.process.compute.function.softmax

import com.wsr.batch.Batch
import com.wsr.batch.elementwise.math.softmax
import com.wsr.batch.elementwise.operation.minus.minus
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class SoftmaxD2 internal constructor(override val outputX: Int, override val outputY: Int) : Compute.D2() {
    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input.softmax()

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input.softmax()
        val delta = calcDelta(output)
        return delta * output * (1f - output)
    }
}

fun <T> NetworkBuilder.D2<T>.softmax() = addProcess(
    process = SoftmaxD2(outputX = inputX, outputY = inputY),
)
