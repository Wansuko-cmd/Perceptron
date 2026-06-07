package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD2ToD3(override val outputX: Int, override val outputY: Int, override val outputZ: Int) :
    Reshape.D2ToD3() {
    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D3> =
        input.reshapeToD3(i = outputX, j = outputY, k = outputZ)

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D2> {
        val output = input.reshapeToD3(i = outputX, j = outputY, k = outputZ)
        val delta = calcDelta(output)
        return delta.reshapeToD2(input.shape)
    }
}

fun <T> NetworkBuilder.D2<T>.reshapeToD3(i: Int = 1, j: Int = inputX, k: Int = inputY) {
    check(i * j * k == inputX * inputY) {
        """
            invalid parameter.
            i: $i
            j: $j
            k: $k
        """.trimIndent()
    }
    addReshape(reshape = ReshapeD2ToD3(outputX = i, outputY = j, outputZ = k))
}
