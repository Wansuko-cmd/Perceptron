#ifndef MAT_MUL_NATIVE_H
#define MAT_MUL_NATIVE_H

#include <stdbool.h>
#include "buffer_native.h"

void com_wsr_cpu_inner(const CPUBuffer* x, const CPUBuffer* y, int b, CPUBuffer* result);

void com_wsr_cpu_mat_mul_d1_to_d2(
    const CPUBuffer* x,
    const CPUBuffer* y, bool trans_y,
    int n, int k,
    CPUBuffer* result
);

void com_wsr_cpu_mat_mul_d2_to_d1(
    const CPUBuffer* x, bool trans_x,
    const CPUBuffer* y,
    int m, int k,
    CPUBuffer* result
);

void com_wsr_cpu_mat_mul_d2_to_d2(
    const CPUBuffer* x, bool trans_x,
    const CPUBuffer* y, bool trans_y,
    int m, int n, int k,
    int b,
    CPUBuffer* result
);

#endif
