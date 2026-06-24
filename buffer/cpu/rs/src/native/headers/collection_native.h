#ifndef COLLECTION_NATIVE_H
#define COLLECTION_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

void com_wsr_cpu_average_d1(const float* x, int size, float* result);

void com_wsr_cpu_average_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_average_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

void com_wsr_cpu_max_d1(const float* x, int size, float* result);

void com_wsr_cpu_max_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_max_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

void com_wsr_cpu_min_d1(const float* x, int size, float* result);

void com_wsr_cpu_min_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_min_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

void com_wsr_cpu_sum_d1(const float* x, int size, float* result);

void com_wsr_cpu_sum_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_sum_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

void com_wsr_cpu_max_index_d1(const float* x, int size, float* result);

void com_wsr_cpu_max_index_d2(const float* x, int xi, int xj, int axis, float* result);

void com_wsr_cpu_max_index_d3(const float* x, int xi, int xj, int xk, int axis, float* result);

void com_wsr_cpu_top_k_d1(const float* x, int size, int k, long long seed, float* result);

void com_wsr_cpu_top_k_d2(const float* x, int xi, int xj, int k, int axis, long long seed, float* result);

void com_wsr_cpu_top_k_d3(const float* x, int xi, int xj, int xk, int k, int axis, long long seed, float* result);

void com_wsr_cpu_top_p_d1(const float* x, int size, float p, long long seed, float* result);

void com_wsr_cpu_top_p_d2(const float* x, int xi, int xj, float p, int axis, long long seed, float* result);

void com_wsr_cpu_top_p_d3(const float* x, int xi, int xj, int xk, float p, int axis, long long seed, float* result);

#ifdef __cplusplus
}
#endif
#endif