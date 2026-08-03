#ifndef INDEX_NATIVE_H
#define INDEX_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

float com_wsr_cpu_gather(const float* x, const float* y, int i, int j, int k, int n, float* result);

float com_wsr_cpu_scatter_add(const float* x, const float* y, int i, int j, int k, int n, int b, float* result);

#ifdef __cplusplus
}
#endif
#endif
