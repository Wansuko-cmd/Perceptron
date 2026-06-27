package com.wsr.knist.network.converter.raw

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class RawD3(override val outputI: Int, override val outputJ: Int, override val outputK: Int) :
    Converter.D3<Batch<IOType.D3>>() {
    override fun encode(input: Batch<IOType.D3>): Batch<IOType.D3> = input
    override fun decode(input: Batch<IOType.D3>): Batch<IOType.D3> = input
}

fun NetworkBuilder.Companion.rawD3(i: Int, j: Int, k: Int, optimizer: Optimizer, initializer: WeightInitializer) =
    inputD3(
        converter = RawD3(i, j, k),
        optimizer = optimizer,
        initializer = initializer,
    )
