package com.wsr.knist.network.process.compute.function.linear

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class LinearD1 internal constructor(override val outputSize: Int) : Compute.D1() {
    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> = input

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> = calcDelta(input)
}

fun <T> NetworkBuilder.D1<T>.linear() = addProcess(LinearD1(outputSize = inputSize))
