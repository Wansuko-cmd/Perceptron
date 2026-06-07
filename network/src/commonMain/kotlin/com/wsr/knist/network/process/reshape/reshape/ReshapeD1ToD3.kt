package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.flatten
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD1ToD3(override val outputX: Int, override val outputY: Int, override val outputZ: Int) : Reshape.D1ToD3() {
    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D3> = input.reshapeToD3(i = outputX, j = outputY, k = outputZ)

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D1> {
        val output = input.reshapeToD3(i = outputX, j = outputY, k = outputZ)
        val delta = calcDelta(output)
        return delta.flatten()
    }
}

fun <T> NetworkBuilder.D1<T>.reshapeToD3(i: Int = 1, j: Int = 1, k: Int = inputSize): NetworkBuilder.D3<T> {
    check(i * j * k == inputSize) {
        """
            invalid parameter.
            i: $i
            j: $j
            k: $k
        """.trimIndent()
    }
    return addReshape(reshape = ReshapeD1ToD3(outputX = i, outputY = j, outputZ = k))
}
