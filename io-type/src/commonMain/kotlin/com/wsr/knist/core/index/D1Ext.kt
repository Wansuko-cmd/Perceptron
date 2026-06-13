package com.wsr.knist.core.index

import com.wsr.knist.Backend
import com.wsr.knist.core.D2
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun IOType.D1.gather(other: IOType.D2): IOType.D2.Global {
    val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
    return IOType.D2(shape = listOf(size, other.j), value = result)
}
