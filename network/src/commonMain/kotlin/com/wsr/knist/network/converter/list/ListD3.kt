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
class LinearD3(override val outputI: Int, override val outputJ: Int, override val outputK: Int) :
    Converter.D3<List<IOType.D3>>() {
    override fun encode(input: List<IOType.D3>): Batch<IOType.D3> = input.toBatch()
    override fun decode(input: Batch<IOType.D3>): List<IOType.D3> = input.toList()
}

fun NetworkBuilder.Companion.listD3(i: Int, j: Int, k: Int, optimizer: Optimizer, initializer: WeightInitializer) =
    inputD3(
        converter = LinearD3(i, j, k),
        optimizer = optimizer,
        initializer = initializer,
    )
