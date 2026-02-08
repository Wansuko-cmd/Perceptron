#ifndef TRANSPOSE_NATIVE_H
#define TRANSPOSE_NATIVE_H

void com_wsr_cpu_transpose_d2(const float* x, int xi, int xj, float* result);

void com_wsr_cpu_transpose_d3(
    const float* x,
    int xi, int xj, int xk,
    int axis_i, int axis_j, int axis_k,
    float* result
);

void com_wsr_cpu_transpose_d4(
    const float* x,
    int xi, int xj, int xk, int xl,
    int axis_i, int axis_j, int axis_k, int axis_l,
    float* result
);
#endif
