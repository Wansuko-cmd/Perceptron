package com.wsr.knist.network.process

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType

interface Process {
    val id: String

    @Suppress("FunctionName")
    fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType>

    @Suppress("FunctionName")
    fun IOScope._train(
        input: Batch<IOType>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
    ): Batch<IOType>

    fun freeze(isFrozen: Boolean) {}
}
