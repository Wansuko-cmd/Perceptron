package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.reshapeToD2
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD2ToD3(
    override val outputI: Int,
    override val outputJ: Int,
    override val outputK: Int,
    override val id: String = Uuid.random().toString(),
) : Reshape.D2ToD3() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D3> =
        input.reshapeToD3(i = outputI, j = outputJ, k = outputK)

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D2> {
        val output = input.reshapeToD3(i = outputI, j = outputJ, k = outputK)
        val delta = calcDelta(output)
        return delta.reshapeToD2(input.shape)
    }
}

fun <T> NetworkBuilder.D2<T>.reshapeToD3(
    i: Int = 1,
    j: Int = inputI,
    k: Int = inputJ,
    id: String = Uuid.random().toString(),
): NetworkBuilder.D3<T> {
    check(i * j * k == inputI * inputJ) {
        """
            invalid parameter.
            i: $i
            j: $j
            k: $k
        """.trimIndent()
    }
    return addReshape(reshape = ReshapeD2ToD3(outputI = i, outputJ = j, outputK = k, id = id))
}
