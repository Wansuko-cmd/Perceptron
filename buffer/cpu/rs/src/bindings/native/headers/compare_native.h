#ifndef COMPARE_NATIVE_H
#define COMPARE_NATIVE_H

#include "buffer_native.h"

#ifdef __cplusplus
extern "C" {
#endif

void com_wsr_cpu_greater_than_d1_to_d0(const CPUBuffer* x, float y, CPUBuffer* result);

void com_wsr_cpu_greater_than_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_less_than_d1_to_d0(const CPUBuffer* x, float y, CPUBuffer* result);

void com_wsr_cpu_less_than_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_equals_d1_to_d0(const CPUBuffer* x, float y, float atol, float rtol, CPUBuffer* result);

void com_wsr_cpu_equals_d1_to_d1(const CPUBuffer* x, const CPUBuffer* y, float atol, float rtol, CPUBuffer* result);

void com_wsr_cpu_where_d0_to_d0(const CPUBuffer* condition, float x, float y, CPUBuffer* result);

void com_wsr_cpu_where_d0_to_d1(const CPUBuffer* condition, float x, const CPUBuffer* y, CPUBuffer* result);

void com_wsr_cpu_where_d1_to_d0(const CPUBuffer* condition, const CPUBuffer* x, float y, CPUBuffer* result);

void com_wsr_cpu_where_d1_to_d1(const CPUBuffer* condition, const CPUBuffer* x, const CPUBuffer* y, CPUBuffer* result);

#ifdef __cplusplus
}
#endif
#endif
