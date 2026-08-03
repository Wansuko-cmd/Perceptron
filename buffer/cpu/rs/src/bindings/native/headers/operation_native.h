#ifndef OPERATION_NATIVE_H
#define OPERATION_NATIVE_H

void com_wsr_cpu_plus_d0_to_d1(
    float x,
    const float* y,
    int y_size,
    float* result
);

void com_wsr_cpu_plus_d1_to_d0(
    const float* x,
    int x_size,
    float y,
    float* result
);

void com_wsr_cpu_plus_d1_to_d1(
    const float* x,
    const float* y,
    int size,
    float* result
);

void com_wsr_cpu_plus_d1_to_d2(
    const float* x,
    const float* y, int yi, int yj,
    int axis,
    float* result
);

void com_wsr_cpu_plus_d1_to_d3(
    const float* x,
    const float* y, int yi, int yj, int yk,
    int axis,
    float* result
);

void com_wsr_cpu_plus_d2_to_d1(
    const float* x, int xi, int xj,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_plus_d2_to_d3(
    const float* x, int xi, int xj,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_plus_d3_to_d1(
    const float* x, int xi, int xj, int xk,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_plus_d3_to_d2(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_plus_d3_to_d4(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_plus_d4_to_d1(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_plus_d4_to_d2(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_plus_d4_to_d3(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_minus_d0_to_d1(
    float x,
    const float* y,
    int y_size,
    float* result
);

void com_wsr_cpu_minus_d1_to_d0(
    const float* x,
    int x_size,
    float y,
    float* result
);

void com_wsr_cpu_minus_d1_to_d1(
    const float* x,
    const float* y,
    int size,
    float* result
);

void com_wsr_cpu_minus_d1_to_d2(
    const float* x,
    const float* y, int yi, int yj,
    int axis,
    float* result
);

void com_wsr_cpu_minus_d1_to_d3(
    const float* x,
    const float* y, int yi, int yj, int yk,
    int axis,
    float* result
);

void com_wsr_cpu_minus_d2_to_d1(
    const float* x, int xi, int xj,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_minus_d2_to_d3(
    const float* x, int xi, int xj,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_minus_d3_to_d1(
    const float* x, int xi, int xj, int xk,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_minus_d3_to_d2(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_minus_d3_to_d4(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_minus_d4_to_d1(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_minus_d4_to_d2(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_minus_d4_to_d3(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_times_d0_to_d1(
    float x,
    const float* y,
    int y_size,
    float* result
);

void com_wsr_cpu_times_d1_to_d0(
    const float* x,
    int x_size,
    float y,
    float* result
);

void com_wsr_cpu_times_d1_to_d1(
    const float* x,
    const float* y,
    int size,
    float* result
);

void com_wsr_cpu_times_d1_to_d2(
    const float* x,
    const float* y, int yi, int yj,
    int axis,
    float* result
);

void com_wsr_cpu_times_d1_to_d3(
    const float* x,
    const float* y, int yi, int yj, int yk,
    int axis,
    float* result
);

void com_wsr_cpu_times_d2_to_d1(
    const float* x, int xi, int xj,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_times_d2_to_d3(
    const float* x, int xi, int xj,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_times_d3_to_d1(
    const float* x, int xi, int xj, int xk,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_times_d3_to_d2(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_times_d3_to_d4(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_times_d4_to_d1(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_times_d4_to_d2(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_times_d4_to_d3(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_div_d0_to_d1(
    float x,
    const float* y,
    int y_size,
    float* result
);

void com_wsr_cpu_div_d1_to_d0(
    const float* x,
    int x_size,
    float y,
    float* result
);

void com_wsr_cpu_div_d1_to_d1(
    const float* x,
    const float* y,
    int size,
    float* result
);

void com_wsr_cpu_div_d1_to_d2(
    const float* x,
    const float* y, int yi, int yj,
    int axis,
    float* result
);

void com_wsr_cpu_div_d1_to_d3(
    const float* x,
    const float* y, int yi, int yj, int yk,
    int axis,
    float* result
);

void com_wsr_cpu_div_d2_to_d1(
    const float* x, int xi, int xj,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_div_d2_to_d3(
    const float* x, int xi, int xj,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_div_d3_to_d1(
    const float* x, int xi, int xj, int xk,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_div_d3_to_d2(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_div_d3_to_d4(
    const float* x, int xi, int xj, int xk,
    const float* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    float* result
);

void com_wsr_cpu_div_d4_to_d1(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y,
    int axis,
    float* result
);

void com_wsr_cpu_div_d4_to_d2(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj,
    int axis1, int axis2,
    float* result
);

void com_wsr_cpu_div_d4_to_d3(
    const float* x, int xi, int xj, int xk, int xl,
    const float* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    float* result
);

#endif
