package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class SwishD3 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val outputK: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input * input.sigmoid()

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val sigmoid = input.sigmoid()
        val output = input * sigmoid
        val delta = calcDelta(output)
        return (output + sigmoid * (1f - output)) * delta
    }
}

fun <T> NetworkBuilder.D3<T>.swish(id: String = Uuid.random().toString()) = addProcess(
    process = SwishD3(outputI = inputI, outputJ = inputJ, outputK = inputK, id = id),
)
