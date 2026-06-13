package com.wsr.knist.network.process.reshape.gad

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.div.div
import com.wsr.knist.batch.reduction.average.average
import com.wsr.knist.batch.shape.broadcastToD2
import com.wsr.knist.batch.shape.broadcastToD3
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class GlobalAverageD3ToD1(private val inputX: Int, private val inputY: Int, private val inputZ: Int) :
    Reshape.D3ToD1() {
    override val outputSize: Int = inputX

    override fun expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D1> = forward(input = input)

    override fun train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D3> {
        val output = forward(input = input)
        val delta = calcDelta(output)
        return (delta / (inputY * inputZ).toFloat())
            .broadcastToD2(axis = 1, size = inputY)
            .broadcastToD3(axis = 2, size = inputZ)
    }

    private fun forward(input: Batch<IOType.D3>) = input
        .reshapeToD2(i = inputX, j = inputY * inputZ)
        .average(axis = 1)
}

fun <T> NetworkBuilder.D3<T>.globalAverageToD1() = addReshape(
    reshape = GlobalAverageD3ToD1(inputX, inputY, inputZ),
)
