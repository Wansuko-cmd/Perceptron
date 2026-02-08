pub fn transpose_d2(x: &[f32], xi: usize, xj: usize, result: &mut [f32]) {
    for (i, d1) in x.chunks_exact(xj).enumerate() {
        for (j, &value) in d1.iter().enumerate() {
            result[j * xi + i] = value;
        }
    }
}

pub fn transpose_d3(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis_i: usize, axis_j: usize, axis_k: usize,
    result: &mut [f32],
) {
    let old_shape = [xi, xj, xk];
    let new_shape = [old_shape[axis_i], old_shape[axis_j], old_shape[axis_k]];

    let old_strides = [xj * xk, xk, 1];

    let ni_stride = old_strides[axis_i];
    let nj_stride = old_strides[axis_j];
    let nk_stride = old_strides[axis_k];

    let mut result_iter = result.iter_mut();

    for n_i in (0..new_shape[0] * ni_stride).step_by(ni_stride) {
        for n_j in (n_i..n_i + new_shape[1] * nj_stride).step_by(nj_stride) {
            for index in (n_j..n_j + new_shape[2] * nk_stride).step_by(nk_stride) {
                if let Some(result_value) = result_iter.next() {
                    *result_value = x[index];
                }
            }
        }
    }
}

pub fn transpose_d4(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
    result: &mut [f32],
) {
    let old_shape = [xi, xj, xk, xl];
    let new_shape = [old_shape[axis_i], old_shape[axis_j], old_shape[axis_k], old_shape[axis_l]];

    let old_strides = [xj * xk * xl, xk * xl, xl, 1];

    let ni_stride = old_strides[axis_i];
    let nj_stride = old_strides[axis_j];
    let nk_stride = old_strides[axis_k];
    let nl_stride = old_strides[axis_l];

    let mut result_iter = result.iter_mut();

    for n_i in (0..new_shape[0] * ni_stride).step_by(ni_stride) {
        for n_j in (n_i..n_i + new_shape[1] * nj_stride).step_by(nj_stride) {
            for n_k in (n_j..n_j + new_shape[2] * nk_stride).step_by(nk_stride) {
                for index in (n_k..n_k + new_shape[3] * nl_stride).step_by(nl_stride) {
                    if let Some(result_value) = result_iter.next() {
                        *result_value = x[index];
                    }
                }
            }
        }
    }
}
