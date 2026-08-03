#ifndef BUFFER_NATIVE_H
#define BUFFER_NATIVE_H

typedef struct CPUBuffer CPUBuffer;

CPUBuffer* com_wsr_cpu_buffer_alloc(int size);
CPUBuffer* com_wsr_cpu_buffer_init(const float* value, int size);
void com_wsr_cpu_buffer_release(CPUBuffer* ptr);
float com_wsr_cpu_buffer_get(const CPUBuffer* ptr, int index);
void com_wsr_cpu_buffer_set(CPUBuffer* ptr, int index, float value);
void com_wsr_cpu_buffer_read_all(const CPUBuffer* ptr, float* result);

#endif
