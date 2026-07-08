package com.wsr.knist.network.process.compute.function.sigmoid

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class SigmoidD3 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val outputK: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input.sigmoid()

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val output = input.sigmoid()
        val delta = calcDelta(output)
        return delta * output * (1f - output)
    }
}

fun <T> NetworkBuilder.D3<T>.sigmoid(id: String = Uuid.random().toString()) = addProcess(
    process = SigmoidD3(outputI = inputI, outputJ = inputJ, outputK = inputK, id = id),
)
