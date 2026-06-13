package com.wsr.knist.batch.index

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.i
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun Batch<IOType.D1>.gather(other: IOType.D2): Batch<IOType.D2.Global> {
    val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
    return Batch.d2(size, i, other.j, result)
}
