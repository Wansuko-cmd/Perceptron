package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD3ToD2(override val outputX: Int, override val outputY: Int) : Reshape.D3ToD2() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D2> =
        input.reshapeToD2(i = outputX, j = outputY)

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D3> {
        val output = input.reshapeToD2(i = outputX, j = outputY)
        val delta = calcDelta(output)
        return delta.reshapeToD3(input.shape)
    }
}

fun <T> NetworkBuilder.D3<T>.reshapeToD2(i: Int, j: Int): NetworkBuilder.D2<T> {
    check(inputX * inputY * inputZ == i * j) {
        """
            invalid parameter.
            inputX: $inputX
            inputY: $inputY
            inputZ: $inputZ
            outputX: $i
            outputY: $j
        """.trimIndent()
    }
    return addReshape(
        reshape = ReshapeD3ToD2(
            outputX = i,
            outputY = j,
        ),
    )
}
