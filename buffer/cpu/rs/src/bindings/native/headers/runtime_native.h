#ifndef BUFFER_RUNTIME_H
#define BUFFER_RUNTIME_H

typedef struct Runtime Runtime;

Runtime* com_wsr_cpu_runtime_alloc();
void com_wsr_cpu_runtime_release(Runtime* runtime);

#endif
