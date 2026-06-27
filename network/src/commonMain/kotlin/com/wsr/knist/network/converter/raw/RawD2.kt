package com.wsr.knist.network.converter.raw

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class RawD2(override val outputI: Int, override val outputJ: Int) : Converter.D2<Batch<IOType.D2>>() {
    override fun encode(input: Batch<IOType.D2>): Batch<IOType.D2> = input
    override fun decode(input: Batch<IOType.D2>): Batch<IOType.D2> = input
}

fun NetworkBuilder.Companion.rawD2(i: Int, j: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD2(
    converter = RawD2(i, j),
    optimizer = optimizer,
    initializer = initializer,
)
