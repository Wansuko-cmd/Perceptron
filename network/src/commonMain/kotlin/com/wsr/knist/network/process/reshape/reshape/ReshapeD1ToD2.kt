package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.flatten
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD1ToD2(override val outputX: Int, override val outputY: Int) : Reshape.D1ToD2() {
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D2> =
        input.reshapeToD2(i = outputX, j = outputY)

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D1> {
        val output = input.reshapeToD2(i = outputX, j = outputY)
        val delta = calcDelta(output)
        return delta.flatten()
    }
}

fun <T> NetworkBuilder.D1<T>.reshapeToD2(i: Int = 1, j: Int = inputSize): NetworkBuilder.D2<T> {
    check(i * j == inputSize) {
        """
            invalid parameter.
            i: $i
            j: $j
        """.trimIndent()
    }
    return addReshape(reshape = ReshapeD1ToD2(outputX = i, outputY = j))
}
