package com.wsr.knist.network.process.reshape.gad

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.reduction.average.average
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class GlobalAverageD3ToD2(private val inputX: Int, private val inputY: Int, private val inputZ: Int) :
    Reshape.D3ToD2() {
    override val outputI: Int = inputY
    override val outputJ: Int = inputZ

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D2> = forward(input)

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D3> {
        val output = forward(input)
        val delta = calcDelta(output)
        return (delta / inputX.toFloat()).broadcastToD3(axis = 0, size = inputX)
    }

    private fun IOScope.forward(input: Batch<IOType.D3>) = input
        .reshapeToD2(i = inputX, j = inputY * inputZ)
        .average(axis = 0)
        .reshapeToD2(i = inputY, j = inputZ)
}

fun <T> NetworkBuilder.D3<T>.globalAverageToD2() = addReshape(
    reshape = GlobalAverageD3ToD2(inputX, inputY, inputZ),
)
