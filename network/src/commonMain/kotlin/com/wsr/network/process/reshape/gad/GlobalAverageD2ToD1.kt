package com.wsr.network.process.reshape.gad

import com.wsr.batch.Batch
import com.wsr.batch.elementwise.operation.div.div
import com.wsr.batch.reduction.average.average
import com.wsr.batch.shape.broadcastToD2
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class GlobalAverageD2ToD1(private val inputX: Int, private val inputY: Int) : Reshape.D2ToD1() {
    override val outputSize: Int = inputX

    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D1> = input.average(axis = 1)

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D2> {
        val output = input.average(axis = 1)
        val delta = calcDelta(output)
        return (delta / inputY.toFloat()).broadcastToD2(1, inputY)
    }
}

fun <T> NetworkBuilder.D2<T>.globalAverageToD1() = addReshape(
    reshape = GlobalAverageD2ToD1(inputX, inputY),
)
