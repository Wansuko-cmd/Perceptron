package com.wsr.knist.core.index

import com.wsr.knist.Backend
import com.wsr.knist.core.D2
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun IOType.D2.scatterAdd(other: IOType.D1, n: Int): IOType.D2.Global {
    val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = j, b = 1)
    return IOType.D2(shape = listOf(n, j), value = result)
}
