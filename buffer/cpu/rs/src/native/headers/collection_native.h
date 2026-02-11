#ifndef COLLECTION_NATIVE_H
#define COLLECTION_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

float com_wsr_cpu_average_d1(const float* x, int size);

void com_wsr_cpu_average_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_average_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

float com_wsr_cpu_max_d1(const float* x, int size);

void com_wsr_cpu_max_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_max_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

float com_wsr_cpu_min_d1(const float* x, int size);

void com_wsr_cpu_min_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_min_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

float com_wsr_cpu_sum_d1(const float* x, int size);

void com_wsr_cpu_sum_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_sum_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

#ifdef __cplusplus
}
#endif
#endif