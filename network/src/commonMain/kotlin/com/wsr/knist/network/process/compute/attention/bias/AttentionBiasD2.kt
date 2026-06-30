package com.wsr.knist.network.process.compute.attention.bias

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.process.Context

sealed interface AttentionBiasD2 {
    fun IOScope.forward(scaled: Batch<IOType.D3>, context: Context): Batch<IOType.D3>
    fun IOScope.backward(delta: Batch<IOType.D3>, context: Context): Batch<IOType.D3>
}
