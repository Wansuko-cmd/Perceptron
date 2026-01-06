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
        for (int j = 0; j < xj; j++) {
            for (int k = 0; k < xk; k++) {
                float acc = x[j * xk + k];
                for (int i = 1; i < xi; i++) {
                    perform_operation<Op>(acc, x[(i * xj + j) * xk + k]);
                }
                result[j * xk + k] = acc;
            }
        }
    } else if (axis == 1) {
        for (int i = 0; i < xi; i++) {
            for (int k = 0; k < xk; k++) {
                float acc = x[i * xj * xk + k];
                for (int j = 1; j < xj; j++) {
                    perform_operation<Op>(acc, x[(i * xj + j) * xk + k]);
                }
                result[i * xk + k] = acc;
            }
        }
    } else if (axis == 2) {
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                float acc = x[(i * xj + j) * xk];
                for (int k = 1; k < xk; k++) {
                    perform_operation<Op>(acc, x[(i * xj + j) * xk + k]);
                }
                result[i * xj + j] = acc;
            }
        }
    }
}

template<Operation Op>
inline void reduce_d4(const float* x, int xi, int xj, int xk, int xl, int axis, float* result) {
    if (axis == 0) {
        for (int j = 0; j < xj; j++) {
            for (int k = 0; k < xk; k++) {
                for (int l = 0; l < xl; l++) {
                    float acc = x[(j * xk + k) * xl + l];
                    for (int i = 1; i < xi; i++) {
                        perform_operation<Op>(acc, x[((i * xj + j) * xk + k) * xl + l]);
                    }
                    result[(j * xk + k) * xl + l] = acc;
                }
            }
        }
    } else if (axis == 1) {
        for (int i = 0; i < xi; i++) {
            for (int k = 0; k < xk; k++) {
                for (int l = 0; l < xl; l++) {
                    float acc = x[(i * xj * xk + k) * xl + l];
                    for (int j = 1; j < xj; j++) {
                        perform_operation<Op>(acc, x[((i * xj + j) * xk + k) * xl + l]);
                    }
                    result[(i * xk + k) * xl + l] = acc;
                }
            }
        }
    } else if (axis == 2) {
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                for (int l = 0; l < xl; l++) {
                    float acc = x[(i * xj + j) * xk * xl + l];
                    for (int k = 1; k < xk; k++) {
                        perform_operation<Op>(acc, x[((i * xj + j) * xk + k) * xl + l]);
                    }
                    result[(i * xj + j) * xl + l] = acc;
                }
            }
        }
    } else if (axis == 3) {
        for (int i = 0; i < xi; i++) {
            for (int j = 0; j < xj; j++) {
                for (int k = 0; k < xk; k++) {
                    float acc = x[((i * xj + j) * xk + k) * xl];
                    for (int l = 1; l < xl; l++) {
                        perform_operation<Op>(acc, x[((i * xj + j) * xk + k) * xl + l]);
                    }
                    result[(i * xj + j) * xk + k] = acc;
                }
            }
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
