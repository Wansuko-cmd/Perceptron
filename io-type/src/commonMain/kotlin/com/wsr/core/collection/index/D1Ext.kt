package com.wsr.core.collection.index

import com.wsr.Backend
import com.wsr.core.IOType
import kotlin.random.Random

fun IOType.D1.topK(k: Int, random: Random = Random): Int = Backend.topK(x = value, k = k, random = random)

fun IOType.D1.topP(p: Float, random: Random = Random): Int = Backend.topP(x = value, p = p, random = random)
