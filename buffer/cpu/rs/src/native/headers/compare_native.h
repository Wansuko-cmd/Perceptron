#ifndef COMPARE_NATIVE_H
#define COMPARE_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

float com_wsr_cpu_greater_than_d1_to_d0(const float* x, int size, float y, float* result);

float com_wsr_cpu_greater_than_d1_to_d1(const float* x, const float* y, int size, float* result);

float com_wsr_cpu_less_than_d1_to_d0(const float* x, int size, float y, float* result);

float com_wsr_cpu_less_than_d1_to_d1(const float* x, const float* y, int size, float* result);

float com_wsr_cpu_where_d0_to_d1(const float* condition, float x, const float* y, int size, float* result);

float com_wsr_cpu_where_d1_to_d0(const float* condition, const float* x, int size, float y, float* result);

float com_wsr_cpu_where_d1_to_d1(const float* condition, const float* x, const float* y, int size, float* result);

#ifdef __cplusplus
}
#endif
#endif
