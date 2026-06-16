package com.wsr.knist.network.converter.linear

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toList
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class LinearD1(override val outputSize: Int) : Converter.D1<IOType.D1>() {
    override fun IOScope.encode(input: List<IOType.D1>): Batch<IOType.D1> = input.toBatch()
    override fun IOScope.decode(input: Batch<IOType.D1>): List<IOType.D1> = input.toList()
}

fun NetworkBuilder.Companion.inputD1(inputSize: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD1(
    converter = LinearD1(inputSize),
    optimizer = optimizer,
    initializer = initializer,
)
