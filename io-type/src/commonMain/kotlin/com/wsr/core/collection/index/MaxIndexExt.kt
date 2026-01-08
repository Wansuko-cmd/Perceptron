package com.wsr.core.collection.index

import com.wsr.Backend
import com.wsr.core.IOType

fun IOType.D1.maxIndex(): Int = Backend.maxIndex(x = value)
