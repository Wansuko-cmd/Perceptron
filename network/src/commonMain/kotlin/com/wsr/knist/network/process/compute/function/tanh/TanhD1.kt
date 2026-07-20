package com.wsr.knist.network.process.compute.function.tanh

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
class TanhD1 internal constructor(override val inputI: Int, override val id: String = Uuid.random().toString()) :
    Compute.D1() {
    override val outputI: Int get() = inputI
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> = input.tanh()

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val output = input.tanh()
        val delta = calcDelta(output)
        return delta * (1f - output.pow(2))
    }
}

fun <T> NetworkBuilder.D1<T>.tanh(id: String = Uuid.random().toString()) = addCompute(
    TanhD1(inputI = inputI, id = id),
)

fun GraphBuilder.D1.tanh(id: String = Uuid.random().toString()) = addCompute(
    TanhD1(inputI = inputI, id = id),
)
