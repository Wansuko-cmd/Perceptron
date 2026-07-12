package com.wsr.knist.network.process

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.optimizer.Optimizer

sealed interface Process {
    val id: String
    val inputShape: List<Int>
    val outputShape: List<Int>

    @Suppress("FunctionName")
    fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType>

    @Suppress("FunctionName")
    fun IOScope._train(
        input: Batch<IOType>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
    ): Batch<IOType>

    fun update(optimizer: Optimizer) {}
}
