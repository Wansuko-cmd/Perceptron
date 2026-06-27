package com.wsr.knist.network.process.reshape.gad

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class GlobalAverageD2ToD1(private val inputX: Int, private val inputY: Int) : Reshape.D2ToD1() {
    override val outputI: Int = inputX

    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D1> = input.average(axis = 1)

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D2> {
        val output = input.average(axis = 1)
        val delta = calcDelta(output)
        return (delta / inputY.toFloat()).broadcastToD2(1, inputY)
    }
}

fun <T> NetworkBuilder.D2<T>.globalAverageToD1() = addReshape(
    reshape = GlobalAverageD2ToD1(inputI, inputJ),
)
