package com.wsr.knist.network.join.add

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.join.Join
import com.wsr.knist.network.process.GraphEnv
import kotlinx.serialization.Serializable

@Serializable
internal class AddD3(override val outputI: Int, override val outputJ: Int, override val outputK: Int) : Join.D3() {
    override fun IOScope.expect(
        inputs: List<Batch<IOType.D3>>,
        env: GraphEnv,
    ): Batch<IOType.D3> = inputs.reduce { acc, batch -> acc + batch }

    override fun IOScope.train(
        inputs: List<Batch<IOType.D3>>,
        env: GraphEnv,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): List<Batch<IOType.D3>> {
        val output = inputs.reduce { acc, batch -> acc + batch }
        val delta = calcDelta(output)
        return List(inputs.size) { delta }
    }
}
