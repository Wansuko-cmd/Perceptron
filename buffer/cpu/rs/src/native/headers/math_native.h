#ifndef MATH_NATIVE_H
#define MATH_NATIVE_H

void com_wsr_cpu_exp_d1(const float* x, float* result, int size);

void com_wsr_cpu_ln_d1(const float* x, float e, float* result, int size);

void com_wsr_cpu_sigmoid_d1(const float* x, float* result, int size);

void com_wsr_cpu_pow_d1(const float* x, int n, float* result, int size);

void com_wsr_cpu_sqrt_d1(const float* x, float e, float* result, int size);

#endif
