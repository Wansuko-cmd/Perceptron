package com.wsr.knist.network.converter.list

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
class ListD1(override val outputI: Int) : Converter.D1<List<IOType.D1>>() {
    override fun encode(input: List<IOType.D1>): Batch<IOType.D1> = input.toBatch()
    override fun decode(input: Batch<IOType.D1>): List<IOType.D1> = input.toList()
}

fun NetworkBuilder.Companion.listD1(i: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD1(
    converter = ListD1(i),
    optimizer = optimizer,
    initializer = initializer,
)
