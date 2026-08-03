#ifndef OPERATION_NATIVE_H
#define OPERATION_NATIVE_H

#include "buffer_native.h"

void com_wsr_cpu_plus_d0_to_d1(float x, const CPUBuffer* y, CPUBuffer* result);
void com_wsr_cpu_plus_d1_to_d0(const CPUBuffer* x, float y, CPUBuffer* result);
void com_wsr_cpu_plus_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_plus_d1_to_d2(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d1_to_d3(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d2_to_d1(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d2_to_d3(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d3_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d3_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d3_to_d4(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d4_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d4_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_plus_d4_to_d3(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d0_to_d1(float x, const CPUBuffer* y, CPUBuffer* result);
void com_wsr_cpu_minus_d1_to_d0(const CPUBuffer* x, float y, CPUBuffer* result);
void com_wsr_cpu_minus_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_minus_d1_to_d2(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d1_to_d3(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d2_to_d1(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d2_to_d3(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d3_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d3_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d3_to_d4(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d4_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d4_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_minus_d4_to_d3(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_times_d0_to_d1(float x, const CPUBuffer* y, CPUBuffer* result);
void com_wsr_cpu_times_d1_to_d0(const CPUBuffer* x, float y, CPUBuffer* result);
void com_wsr_cpu_times_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_times_d1_to_d2(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_times_d1_to_d3(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_times_d2_to_d1(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_times_d2_to_d3(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_times_d3_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_times_d3_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_times_d3_to_d4(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_times_d4_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_times_d4_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_times_d4_to_d3(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_div_d0_to_d1(float x, const CPUBuffer* y, CPUBuffer* result);
void com_wsr_cpu_div_d1_to_d0(const CPUBuffer* x, float y, CPUBuffer* result);
void com_wsr_cpu_div_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_div_d1_to_d2(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_div_d1_to_d3(
    const CPUBuffer* x,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_div_d2_to_d1(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_div_d2_to_d3(
    const CPUBuffer* x, int xi, int xj,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_div_d3_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_div_d3_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_div_d3_to_d4(
    const CPUBuffer* x, int xi, int xj, int xk,
    const CPUBuffer* y, int yi, int yj, int yk, int yl,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

void com_wsr_cpu_div_d4_to_d1(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y,
    int axis,
    CPUBuffer* result
);

void com_wsr_cpu_div_d4_to_d2(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj,
    int axis1, int axis2,
    CPUBuffer* result
);

void com_wsr_cpu_div_d4_to_d3(
    const CPUBuffer* x, int xi, int xj, int xk, int xl,
    const CPUBuffer* y, int yi, int yj, int yk,
    int axis1, int axis2, int axis3,
    CPUBuffer* result
);

#endif
