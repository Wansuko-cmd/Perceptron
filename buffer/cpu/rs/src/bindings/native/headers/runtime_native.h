#ifndef BUFFER_RUNTIME_H
#define BUFFER_RUNTIME_H

typedef struct Runtime Runtime;

Runtime* com_wsr_cpu_runtime_allocate(long pool_size);
void com_wsr_cpu_runtime_release(Runtime* runtime);

#endif
