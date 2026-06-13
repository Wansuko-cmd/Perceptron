package com.wsr.knist.core.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.core.D0
import com.wsr.knist.core.IOType
import com.wsr.knist.core.get
import com.wsr.knist.scope.ScopeOp
import kotlin.math.pow

@ScopeOp
fun IOType.D0.pow(n: Int): IOType.D0.Global {
    val result = Backend.pow(x = value, n = n)
    return IOType.D0(value = result)
}

@ScopeOp
fun IOType.D0.sqrt(e: Float = 1e-7f): IOType.D0.Global {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D0(value = result)
}
