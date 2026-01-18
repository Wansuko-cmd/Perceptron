#ifndef MATH_FUN_H
#define MATH_FUN_H

#ifdef __cplusplus
extern "C" {
#endif

void exp_d1(const float* x, float* result, int size);

void ln_d1(const float* x, float e, float* result, int size);

void pow_d1(const float* x, int n, float* result, int size);

void sqrt_d1(const float* x, float e, float* result, int size);

#ifdef __cplusplus
}
#endif
#endif
