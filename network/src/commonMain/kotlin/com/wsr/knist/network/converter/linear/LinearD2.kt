package com.wsr.knist.network.converter.linear

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toList
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class LinearD2(override val outputI: Int, override val outputJ: Int) : Converter.D2<IOType.D2>() {
    override fun encode(input: List<IOType.D2>): Batch<IOType.D2> = input.toBatch()
    override fun decode(input: Batch<IOType.D2>): List<IOType.D2> = input.toList()
}

fun NetworkBuilder.Companion.inputD2(x: Int, y: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD2(
    converter = LinearD2(x, y),
    optimizer = optimizer,
    initializer = initializer,
)
