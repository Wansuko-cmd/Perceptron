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

    let mut result_iter = result.iter_mut();
    for ni in 0..new_shape[0] {
        for nj in 0..new_shape[1] {
            for nk in 0..new_shape[2] {
                let mut old_indexes = [0, 0, 0];
                old_indexes[axis_i] = ni;
                old_indexes[axis_j] = nj;
                old_indexes[axis_k] = nk;
                let old_index = (old_indexes[0] * xj + old_indexes[1]) * xk + old_indexes[2];

                if let Some(result_value) = result_iter.next() {
                    *result_value = x[old_index];
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

    let mut result_iter = result.iter_mut();
    for ni in 0..new_shape[0] {
        for nj in 0..new_shape[1] {
            for nk in 0..new_shape[2] {
                for nl in 0..new_shape[3] {
                    let mut old_indexes = [0, 0, 0, 0];
                    old_indexes[axis_i] = ni;
                    old_indexes[axis_j] = nj;
                    old_indexes[axis_k] = nk;
                    old_indexes[axis_l] = nl;
                    let old_index = ((old_indexes[0] * xj + old_indexes[1]) * xk + old_indexes[2]) * xl + old_indexes[3];

                    if let Some(result_value) = result_iter.next() {
                        *result_value = x[old_index];
                    }
                }
            }
        }
    }
}
