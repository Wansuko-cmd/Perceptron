package com.wsr.knist.network.process.reshape.reshape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.flatten
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
internal class ReshapeD3ToD1(override val outputI: Int, override val id: String = Uuid.random().toString()) :
    Reshape.D3ToD1() {
    constructor(inputI: Int, inputJ: Int, inputK: Int, id: String = Uuid.random().toString()) : this(
        outputI = inputI * inputJ * inputK,
        id = id,
    )

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D1> = input.flatten()

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D3> {
        val output = input.flatten()
        val delta = calcDelta(output)
        return delta.reshapeToD3(input.shape)
    }
}

fun <T> NetworkBuilder.D3<T>.reshapeToD1(id: String = Uuid.random().toString()): NetworkBuilder.D1<T> = addReshape(
    reshape = ReshapeD3ToD1(inputI = inputI, inputJ = inputJ, inputK = inputK, id = id),
)
