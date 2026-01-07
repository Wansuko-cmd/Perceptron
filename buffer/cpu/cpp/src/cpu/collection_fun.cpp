#include <collection_fun.h>
#include <cfloat>
#include <algorithm>

enum class Operation {
    Max,
    Min,
    Sum
};

template<Operation Op>
inline void initialize(float* result, int size) {
    if constexpr (Op == Operation::Max) {
        std::fill(result, result + size, -FLT_MAX);
    } else if constexpr (Op == Operation::Min) {
        std::fill(result, result + size, FLT_MAX);
    } else if constexpr (Op == Operation::Sum) {
        std::fill(result, result + size, 0.0f);
    }
}

template<Operation Op>
inline void perform_operation(float& acc, float x) {
    if constexpr (Op == Operation::Max) {
        acc = std::max(acc, x);
    } else if constexpr (Op == Operation::Min) {
        acc = std::min(acc, x);
    } else if constexpr (Op == Operation::Sum) {
        acc += x;
    }
}

template<Operation Op>
inline float reduce_d1(const float* x, size_t size) {
    float acc = x[0];
    for (size_t i = 1; i < size; i++) {
        perform_operation<Op>(acc, x[i]);
    }
    return acc;
}

template<Operation Op>
inline void reduce_d2(const float* x, int xi, int xj, int axis, float* result) {
    if (axis == 0) {
        initialize<Op>(result, xj);
        for (int i = 0; i < xi; i++) {
            const float* x_view = x + i * xj;
            for (int j = 0; j < xj; j++) {
                perform_operation<Op>(result[j], x_view[j]);
            }
        }
    } else if (axis == 1) {
        initialize<Op>(result, xi);
        for (int i = 0; i < xi; i++) {
            const float* x_view = x + i * xj;
            for (int j = 0; j < xj; j++) {
                perform_operation<Op>(result[i], x_view[j]);
            }
        }
    }
}

template<Operation Op>
inline void reduce_d3(const float* x, int xi, int xj, int xk, int axis, float* result) {
    if (axis == 0) {
        initialize<Op>(result, xj * xk);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                float* result_view = result + j * xk;
                const float* x_view = x + (i * xj + j) * xk;
                for (int k = 0; k < xk; k++) {
                    perform_operation<Op>(result_view[k], x_view[k]);
                }
            }
        }
    } else if (axis == 1) {
        initialize<Op>(result, xi * xk);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                float* result_view = result + i * xk;
                const float* x_view = x + (i * xj + j) * xk;
                for (int k = 0; k < xk; k++) {
                    perform_operation<Op>(result_view[k], x_view[k]);
                }
            }
        }
    } else if (axis == 2) {
        initialize<Op>(result, xi * xj);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                float* result_view = result + i * xj + j;
                const float* x_view = x + (i * xj + j) * xk;
                for (int k = 0; k < xk; k++) {
                    perform_operation<Op>(*result_view, x_view[k]);
                }
            }
        }
    }
}

template<Operation Op>
inline void reduce_d4(const float* x, int xi, int xj, int xk, int xl, int axis, float* result) {
    if (axis == 0) {
        initialize<Op>(result, xj * xk * xl);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                for (int k = 0; k < xk; k++) {
                    float* result_view = result + (j * xk + k) * xl;
                    const float* x_view = x + ((i * xj + j) * xk + k) * xl;
                    for (int l = 0; l < xl; l++) {
                        perform_operation<Op>(result_view[l], x_view[l]);
                    }
                }
            }
        }
    } else if (axis == 1) {
        initialize<Op>(result, xi * xk * xl);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                for (int k = 0; k < xk; k++) {
                    float* result_view = result + (i * xk + k) * xl;
                    const float* x_view = x + ((i * xj + j) * xk + k) * xl;
                    for (int l = 0; l < xl; l++) {
                        perform_operation<Op>(result_view[l], x_view[l]);
                    }
                }
            }
        }
    } else if (axis == 2) {
        initialize<Op>(result, xi * xj * xl);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                for (int k = 0; k < xk; k++) {
                    float* result_view = result + (i * xj + j) * xl;
                    const float* x_view = x + ((i * xj + j) * xk + k) * xl;
                    for (int l = 0; l < xl; l++) {
                        perform_operation<Op>(result_view[l], x_view[l]);
                    }
                }
            }
        }
    } else if (axis == 3) {
        initialize<Op>(result, xi * xj * xk);
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                for (int k = 0; k < xk; k++) {
                    float* result_view = result + (i * xj + j) * xk + k;
                    const float* x_view = x + ((i * xj + j) * xk + k) * xl;
                    for (int l = 0; l < xl; l++) {
                        perform_operation<Op>(*result_view, x_view[l]);
                    }
                }
            }
        }
    }
}

float average_d1(const float* x, size_t size) {
    return reduce_d1<Operation::Sum>(x, size) / size;
}

void average_d2(const float* x, int xi, int xj, int axis, float* result) {
    reduce_d2<Operation::Sum>(x, xi, xj, axis, result);
    if (axis == 0) {
        float inv = 1.0f / xi;
        for (int n = 0; n < xj; n++) {
            result[n] *= inv;
        }
    } else {
        float inv = 1.0f / xj;
        for (int n = 0; n < xi; n++) {
            result[n] *= inv;
        }
    }
}

void average_d3(const float* x, int xi, int xj, int xk, int axis, float* result) {
    reduce_d3<Operation::Sum>(x, xi, xj, xk, axis, result);
    if (axis == 0) {
        float inv = 1.0f / xi;
        for (int n = 0; n < xj * xk; n++) {
            result[n] *= inv;
        }
    } else if (axis == 1) {
        float inv = 1.0f / xj;
        for (int n = 0; n < xi * xk; n++) {
            result[n] *= inv;
        }
    } else {
        float inv = 1.0f / xk;
        for (int n = 0; n < xi * xj; n++) {
            result[n] *= inv;
        }
    }
}

void average_d4(const float* x, int xi, int xj, int xk, int xl, int axis, float* result) {
    reduce_d4<Operation::Sum>(x, xi, xj, xk, xl, axis, result);
    if (axis == 0) {
        float inv = 1.0f / xi;
        for (int n = 0; n < xj * xk * xl; n++) {
            result[n] *= inv;
        }
    } else if (axis == 1) {
        float inv = 1.0f / xj;
        for (int n = 0; n < xi * xk * xl; n++) {
            result[n] *= inv;
        }
    } else if (axis == 2) {
        float inv = 1.0f / xk;
        for (int n = 0; n < xi * xj * xl; n++) {
            result[n] *= inv;
        }
    } else {
        float inv = 1.0f / xl;
        for (int n = 0; n < xi * xj * xk; n++) {
            result[n] *= inv;
        }
    }
}

float max_d1(const float* x, size_t size) {
    return reduce_d1<Operation::Max>(x, size);
}

void max_d2(const float* x, int xi, int xj, int axis, float* result) {
    reduce_d2<Operation::Max>(x, xi, xj, axis, result);
}

void max_d3(const float* x, int xi, int xj, int xk, int axis, float* result) {
    reduce_d3<Operation::Max>(x, xi, xj, xk, axis, result);
}

void max_d4(const float* x, int xi, int xj, int xk, int xl, int axis, float* result) {
    reduce_d4<Operation::Max>(x, xi, xj, xk, xl, axis, result);
}

float min_d1(const float* x, size_t size) {
    return reduce_d1<Operation::Min>(x, size);
}

void min_d2(const float* x, int xi, int xj, int axis, float* result) {
    reduce_d2<Operation::Min>(x, xi, xj, axis, result);
}

void min_d3(const float* x, int xi, int xj, int xk, int axis, float* result) {
    reduce_d3<Operation::Min>(x, xi, xj, xk, axis, result);
}

void min_d4(const float* x, int xi, int xj, int xk, int xl, int axis, float* result) {
    reduce_d4<Operation::Min>(x, xi, xj, xk, xl, axis, result);
}

float sum_d1(const float* x, size_t size) {
    return reduce_d1<Operation::Sum>(x, size);
}

void sum_d2(const float* x, int xi, int xj, int axis, float* result) {
    reduce_d2<Operation::Sum>(x, xi, xj, axis, result);
}

void sum_d3(const float* x, int xi, int xj, int xk, int axis, float* result) {
    reduce_d3<Operation::Sum>(x, xi, xj, xk, axis, result);
}

void sum_d4(const float* x, int xi, int xj, int xk, int xl, int axis, float* result) {
    reduce_d4<Operation::Sum>(x, xi, xj, xk, xl, axis, result);
}
