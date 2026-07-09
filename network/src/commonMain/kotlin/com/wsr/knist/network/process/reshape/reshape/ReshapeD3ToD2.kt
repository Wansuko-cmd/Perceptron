package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Reshape
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD3ToD2(
    override val inputI: Int,
    override val inputJ: Int,
    override val inputK: Int,
    override val outputI: Int,
    override val outputJ: Int,
    override val id: String = Uuid.random().toString(),
) : Reshape.D3ToD2() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D2> =
        input.reshapeToD2(i = outputI, j = outputJ)

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D3> {
        val output = input.reshapeToD2(i = outputI, j = outputJ)
        val delta = calcDelta(output)
        return delta.reshapeToD3(input.shape)
    }
}

fun <T> NetworkBuilder.D3<T>.reshapeToD2(i: Int, j: Int, id: String = Uuid.random().toString()): NetworkBuilder.D2<T> {
    check(inputI * inputJ * inputK == i * j) {
        """
            invalid parameter.
            inputI: $inputI
            inputJ: $inputJ
            inputK: $inputK
            outputI: $i
            outputJ: $j
        """.trimIndent()
    }
    return addReshape(
        reshape = ReshapeD3ToD2(
            inputI = inputI,
            inputJ = inputJ,
            inputK = inputK,
            outputI = i,
            outputJ = j,
            id = id,
        ),
    )
}
