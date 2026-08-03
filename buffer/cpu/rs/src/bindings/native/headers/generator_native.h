#ifndef GENERATOR_NATIVE_H
#define GENERATOR_NATIVE_H

#include "buffer_native.h"

#ifdef __cplusplus
extern "C" {
#endif

void com_wsr_cpu_random_d1(float from, float until, long long seed, CPUBuffer* result);

#ifdef __cplusplus
}
#endif
#endif
