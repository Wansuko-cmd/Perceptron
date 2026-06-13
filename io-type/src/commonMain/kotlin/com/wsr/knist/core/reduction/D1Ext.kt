package com.wsr.knist.core.reduction

import com.wsr.knist.Backend
import com.wsr.knist.core.D0
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.random.Random

@ScopeOp
fun IOType.D1.average(): IOType.D0.Global = IOType.D0(Backend.average(value))

@ScopeOp
fun IOType.D1.max(): IOType.D0.Global = IOType.D0(Backend.max(x = value))

@ScopeOp
fun IOType.D1.min() = IOType.D0(Backend.min(x = value))

@ScopeOp
fun IOType.D1.sum(): IOType.D0.Global = IOType.D0(Backend.sum(x = value))

@ScopeOp
fun IOType.D1.topK(k: Int, random: Random = Random): IOType.D0.Global =
    IOType.D0(Backend.topK(x = value, k = k, random = random))

@ScopeOp
fun IOType.D1.topP(p: Float, random: Random = Random): IOType.D0.Global =
    IOType.D0(Backend.topP(x = value, p = p, random = random))

@ScopeOp
fun IOType.D1.maxIndex(): IOType.D0.Global = IOType.D0(Backend.maxIndex(x = value))
