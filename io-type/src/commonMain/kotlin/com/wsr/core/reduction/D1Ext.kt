package com.wsr.core.reduction

import com.wsr.Backend
import com.wsr.core.IOType
import kotlin.random.Random

fun IOType.D1.average(): IOType.D0 = IOType.D0(Backend.average(value))

fun IOType.D1.max(): IOType.D0 = IOType.D0(Backend.max(x = value))

fun IOType.D1.min() = IOType.D0(Backend.min(x = value))

fun IOType.D1.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value))

fun IOType.D1.topK(k: Int, random: Random = Random): Int = Backend.topK(x = value, k = k, random = random)

fun IOType.D1.topP(p: Float, random: Random = Random): Int = Backend.topP(x = value, p = p, random = random)

fun IOType.D1.maxIndex(): Int = Backend.maxIndex(x = value)
