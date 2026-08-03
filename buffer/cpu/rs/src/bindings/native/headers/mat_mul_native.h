#ifndef MAT_MUL_NATIVE_H
#define MAT_MUL_NATIVE_H

#include <stdbool.h>

void com_wsr_cpu_inner(float* x, float* y, int size, int b, float* result);

void com_wsr_cpu_mat_mul_d1_to_d2(
    const float* x,
    const float* y, bool trans_y,
    int n, int k,
    float* result
);

void com_wsr_cpu_mat_mul_d2_to_d1(
    const float* x, bool trans_x,
    const float* y,
    int m, int k,
    float* result
);

void com_wsr_cpu_mat_mul_d2_to_d2(
    float* x, bool trans_x,
    float* y, bool trans_y,
    int m, int n, int k,
    int b,
    float* result
);

#endif
