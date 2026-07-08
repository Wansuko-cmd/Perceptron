package com.wsr.knist.network.process.compute.scale.d2

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class ScaleD2 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2.Global,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input * weight

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input * weight
        val delta = calcDelta(output)

        val dx = delta * weight
        weight = optimizer.adapt(
            weight = weight,
            dw = input * delta,
        ).toGlobal()

        return dx
    }

    override fun freeze(isFrozen: Boolean) {
        optimizer.isFrozen = isFrozen
    }
}

fun <T> NetworkBuilder.D2<T>.scale(
    axis: Int? = null,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(1f),
    id: String = Uuid.random().toString(),
): NetworkBuilder.D2<T> {
    val process = when (axis) {
        null -> ScaleD2(
            outputI = inputI,
            outputJ = inputJ,
            optimizer = optimizer.d2(
                inputI,
                inputJ,
            ),
            weight = initializer.d2(
                input = listOf(inputI, inputJ),
                output = listOf(inputI, inputJ),
                i = inputI,
                j = inputJ,
            ),
            id = id,
        )

        0, 1 -> {
            val inputT = if (axis == 0) inputI else inputJ
            ScaleAxisD2(
                outputI = inputI,
                outputJ = inputJ,
                axis = axis,
                optimizer = optimizer.d1(inputT),
                weight = initializer.d1(
                    input = listOf(inputT),
                    output = listOf(inputT),
                    size = inputT,
                ),
                id = id,
            )
        }

        else -> throw IllegalStateException(
            """
            invalid parameter.
            axis: $axis
            """.trimIndent(),
        )
    }
    return addProcess(process = process)
}
