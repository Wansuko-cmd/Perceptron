package com.wsr.knist.network.join.add

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.join.Join
import com.wsr.knist.network.process.GraphEnv
import kotlinx.serialization.Serializable

@Serializable
internal class AddD1(override val outputI: Int) : Join.D1() {
    override fun IOScope.expect(
        inputs: List<Batch<IOType.D1>>,
        env: GraphEnv,
    ): Batch<IOType.D1> = inputs.reduce { acc, batch -> acc + batch }

    override fun IOScope.train(
        inputs: List<Batch<IOType.D1>>,
        env: GraphEnv,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): List<Batch<IOType.D1>> {
        val output = inputs.reduce { acc, batch -> acc + batch }
        val delta = calcDelta(output)
        return List(inputs.size) { delta }
    }
}
