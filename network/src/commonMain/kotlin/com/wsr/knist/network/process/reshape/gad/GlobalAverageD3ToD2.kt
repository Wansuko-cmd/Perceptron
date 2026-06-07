package com.wsr.knist.network.process.reshape.gad

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.div.div
import com.wsr.knist.batch.reduction.average.average
import com.wsr.knist.batch.shape.broadcastToD3
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toList
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.elementwise.operation.div.div
import com.wsr.knist.core.get
import com.wsr.knist.core.shape.transpose
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class GlobalAverageD3ToD2(private val inputX: Int, private val inputY: Int, private val inputZ: Int) :
    Reshape.D3ToD2() {
    override val outputX: Int = inputY
    override val outputY: Int = inputZ

    override fun expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D2> = forward(input)

    override fun train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D3> {
        val output = forward(input)
        val delta = calcDelta(output)
        return (delta / inputX.toFloat()).broadcastToD3(axis = 0, size = inputX)
    }

    private fun forward(input: Batch<IOType.D3>) = input
        .reshapeToD2(i = inputX, j = inputY * inputZ)
        .average(axis = 0)
        .reshapeToD2(i = inputY, j = inputZ)
}

fun <T> NetworkBuilder.D3<T>.globalAverageToD2() = addReshape(
    reshape = GlobalAverageD3ToD2(inputX, inputY, inputZ),
)
