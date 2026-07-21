package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphScope.addCompute
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.process.Context
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class SwishD2 internal constructor(
    override val inputI: Int,
    override val inputJ: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override val outputI: Int get() = inputI
    override val outputJ: Int get() = inputJ
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input * input.sigmoid()

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val sigmoid = input.sigmoid()
        val output = input * sigmoid
        val delta = calcDelta(output)
        return (output + sigmoid * (1f - output)) * delta
    }
}

fun <T> NetworkBuilder.D2<T>.swish(id: String = Uuid.random().toString()) = addCompute(
    SwishD2(inputI = inputI, inputJ = inputJ, id = id),
)

fun GraphBuilder.Node.D2.swish(id: String = Uuid.random().toString()) = addCompute(
    SwishD2(inputI = inputI, inputJ = inputJ, id = id),
)
