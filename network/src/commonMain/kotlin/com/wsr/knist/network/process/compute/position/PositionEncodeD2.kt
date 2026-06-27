package com.wsr.knist.network.process.compute.position

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlinx.serialization.Serializable

@Serializable
class PositionEncodeD2 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    private val waveLength: Float,
) : Compute.D2() {
    private val position by lazy {
        IOType.d2(outputI, outputJ) { x, y ->
            if (y % 2 == 0) {
                sin(x / waveLength.pow(y / outputJ.toFloat()))
            } else {
                cos(x / waveLength.pow((y - 1) / outputJ.toFloat()))
            }
        }
    }

    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input + position

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input + position
        return calcDelta(output)
    }
}

fun <T> NetworkBuilder.D2<T>.positionEncode(waveLength: Float = 10000f) = addProcess(
    process = PositionEncodeD2(
        outputI = inputI,
        outputJ = inputJ,
        waveLength = waveLength,
    ),
)
