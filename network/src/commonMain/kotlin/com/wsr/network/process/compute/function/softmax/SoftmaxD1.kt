package com.wsr.network.process.compute.function.softmax

import com.wsr.batch.Batch
import com.wsr.batch.math.softmax
import com.wsr.batch.operation.minus.minus
import com.wsr.batch.operation.times.times
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class SoftmaxD1 internal constructor(override val outputSize: Int) : Compute.D1() {
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> = input.softmax()

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val output = input.softmax()
        val delta = calcDelta(output)
        return delta * output * (1f - output)
    }
}

fun <T> NetworkBuilder.D1<T>.softmax() = addProcess(SoftmaxD1(outputSize = inputSize))
