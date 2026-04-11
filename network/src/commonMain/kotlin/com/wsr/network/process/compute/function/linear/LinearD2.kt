package com.wsr.network.process.compute.function.linear

import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class LinearD2 internal constructor(override val outputX: Int, override val outputY: Int) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> = calcDelta(input)
}

fun <T> NetworkBuilder.D2<T>.linear() = addProcess(
    process = LinearD2(outputX = inputX, outputY = inputY),
)
