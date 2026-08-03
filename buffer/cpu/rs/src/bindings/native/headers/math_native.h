#ifndef MATH_NATIVE_H
#define MATH_NATIVE_H

#include "buffer_native.h"

void com_wsr_cpu_exp_d1(const CPUBuffer* x, CPUBuffer* result);

void com_wsr_cpu_ln_d1(const CPUBuffer* x, float e, CPUBuffer* result);

void com_wsr_cpu_sigmoid_d1(const CPUBuffer* x, CPUBuffer* result);

void com_wsr_cpu_pow_d1(const CPUBuffer* x, int n, CPUBuffer* result);

void com_wsr_cpu_sqrt_d1(const CPUBuffer* x, float e, CPUBuffer* result);

#endif
