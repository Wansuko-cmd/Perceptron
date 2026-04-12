package com.wsr.network.process

import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.scope.IOScope

interface Process {
    @Suppress("FunctionName")
    fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType>

    @Suppress("FunctionName")
    fun IOScope._train(input: Batch<IOType>, context: Context, calcDelta: (Batch<IOType>) -> Batch<IOType>): Batch<IOType>
}
