package com.wsr.knist.network.process

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType

interface Process {
    @Suppress("FunctionName")
    fun _expect(input: Batch<IOType>, context: Context): Batch<IOType>

    @Suppress("FunctionName")
    fun _train(input: Batch<IOType>, context: Context, calcDelta: (Batch<IOType>) -> Batch<IOType>): Batch<IOType>
}
