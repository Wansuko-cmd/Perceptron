package com.wsr.knist.network.converter.raw

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toBatch
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

fun NetworkBuilder.Companion.inputD3(x: Int, y: Int, z: Int, optimizer: Optimizer, initializer: WeightInitializer) =
    inputD3(
        converter = RawD3(x, y, z),
        optimizer = optimizer,
        initializer = initializer,
    )
