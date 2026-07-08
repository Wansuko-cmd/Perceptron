package com.wsr.knist.network.process.compute.debug

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

@Serializable
class DebugD3 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val outputK: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    @Transient
    var onInput: (Batch<IOType.D3>) -> Unit = {}

    @Transient
    var onDelta: (Batch<IOType.D3>) -> Unit = {}

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input.also {
        onInput(it)
    }

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val input = input.also { onInput(it) }
        val delta = calcDelta(input).also { onDelta(it) }
        return delta
    }
}

/**
 * ※Json化するとラムダ式はリセットされる
 */
fun <T> NetworkBuilder.D3<T>.debug(
    onInput: (Batch<IOType.D3>) -> Unit = {},
    onDelta: (Batch<IOType.D3>) -> Unit = {},
    id: String = Uuid.random().toString(),
) = addProcess(
    process = DebugD3(
        outputI = inputI,
        outputJ = inputJ,
        outputK = inputK,
        id = id,
    ).apply {
        this.onInput = onInput
        this.onDelta = onDelta
    },
)
