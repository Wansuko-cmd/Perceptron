package com.wsr.knist.network.optimizer.freeze

import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
data object Freeze : Optimizer {
    override fun d1(i: Int): Optimizer.D1 = FreezeD1

    override fun d2(i: Int, j: Int): Optimizer.D2 = FreezeD2

    override fun d3(i: Int, j: Int, k: Int): Optimizer.D3 = FreezeD3

    override fun d4(i: Int, j: Int, k: Int, l: Int): Optimizer.D4 = FreezeD4
}

@Serializable
internal data object FreezeD1 : Optimizer.D1() {
    override fun IOScope.adapt(weight: IOType.D1, dw: IOType.D1): IOType.D1 = weight
}

@Serializable
internal data object FreezeD2 : Optimizer.D2() {
    override fun IOScope.adapt(weight: IOType.D2, dw: IOType.D2): IOType.D2 = weight
}

@Serializable
internal data object FreezeD3 : Optimizer.D3() {
    override fun IOScope.adapt(weight: IOType.D3, dw: IOType.D3): IOType.D3 = weight
}

@Serializable
internal data object FreezeD4 : Optimizer.D4() {
    override fun IOScope.adapt(weight: IOType.D4, dw: IOType.D4): IOType.D4 = weight
}
