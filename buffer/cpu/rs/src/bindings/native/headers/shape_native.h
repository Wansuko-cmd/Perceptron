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

void com_wsr_cpu_transpose_d5(
    const float* x,
    int xi, int xj, int xk, int xl, int xm,
    int axis_i, int axis_j, int axis_k, int axis_l, int axis_m,
    float* result
);

void com_wsr_cpu_slice_d1(
    const float* x, int x_size,
    int start, int end, int step,
    float* result, int result_size
);

void com_wsr_cpu_slice_d2(
    const float* x,
    int xi, int xj, int axis,
    int start, int end, int step,
    float* result, int result_size
);

void com_wsr_cpu_slice_d3(
    const float* x,
    int xi, int xj, int xk, int axis,
    int start, int end, int step,
    float* result, int result_size
);

void com_wsr_cpu_copy_into_d1(
    const float* x, int x_size,
    float* result, int result_size,
    int start, int end, int step
);

void com_wsr_cpu_copy_into_d2(
    const float* x, int x_size,
    float* result,
    int ri, int rj, int axis,
    int start, int end, int step
);

void com_wsr_cpu_copy_into_d3(
    const float* x, int x_size,
    float* result,
    int ri, int rj, int rk, int axis,
    int start, int end, int step
);

void com_wsr_cpu_unfold_d1(
    const float* x,
    int xi, int xj,
    int b,
    int window, int stride, int dilation, int padding,
    float* result
);

void com_wsr_cpu_unfold_d2(
    const float* x,
    int xi, int xj, int xk,
    int b,
    int window, int stride, int dilation, int padding,
    float* result
);

void com_wsr_cpu_fold_d1(
    const float* x,
    int xi, int xj, int xk,
    int b,
    int stride, int dilation, int padding,
    float* result
);

void com_wsr_cpu_fold_d2(
    const float* x,
    int xi, int xj, int xk, int xl,
    int b,
    int stride, int dilation, int padding,
    float* result
);

void com_wsr_cpu_flip_d3(
    const float* x,
    int xi, int xj, int xk,
    int axis,
    float* result
);

#endif
