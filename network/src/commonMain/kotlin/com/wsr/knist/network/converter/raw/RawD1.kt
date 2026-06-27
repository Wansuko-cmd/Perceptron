package com.wsr.knist.network.converter.raw

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class RawD1(override val outputI: Int) : Converter.D1<Batch<IOType.D1>>() {
    override fun encode(input: Batch<IOType.D1>): Batch<IOType.D1> = input
    override fun decode(input: Batch<IOType.D1>): Batch<IOType.D1> = input
}

fun NetworkBuilder.Companion.rawD1(i: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD1(
    converter = RawD1(i),
    optimizer = optimizer,
    initializer = initializer,
)
