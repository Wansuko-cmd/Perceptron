pub fn transpose_d2(x: &[f32], xi: usize, xj: usize, result: &mut [f32]) {
    let mut result_iter = result.iter_mut();

    for n_i in 0..xj {
        for index in (n_i..n_i + xi * xj).step_by(xj) {
            if let Some(result_value) = result_iter.next() {
                *result_value = x[index];
            }
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

pub fn slice_d1(x: &[f32], start: usize, end: usize, step: isize, result: &mut [f32]) {
    match step {
        0 => panic!("invalid parameter. [step: {}]", step),
        1 => {
            let size = result.len().min(end - start + 1);
            result[..size].copy_from_slice(&x[start..start + size]);
        }
        _ => {
            let indices = create_indices(start, end, step);
            for (i, res) in indices.iter().zip(result.iter_mut()) {
                *res = x[*i];
            }
        }
    }
}

pub fn slice_d2(x: &[f32], xi: usize, xj: usize, axis: usize, start: usize, end: usize, step: isize, result: &mut [f32]) {
    match step {
        0 => panic!("invalid parameter. [axis: {}, step: {}]", axis, step),
        _ => {
            match axis {
                0 => {
                    let indices = create_indices(start, end, step);
                    for (r_i, x_i) in indices.iter().enumerate() {
                        let x_offset = x_i * xj;
                        let result_offset = r_i * xj;
                        result[result_offset..result_offset + xj].copy_from_slice(&x[x_offset..x_offset + xj]);
                    }
                }
                1 => {
                    let indices = create_indices(start, end, step);
                    let size = indices.len();
                    for i in 0..xi {
                        let x_offset = i * xj;
                        let result_offset = i * size;
                        for (r_j, x_j) in indices.iter().enumerate() {
                            result[result_offset + r_j] = x[x_offset + x_j];
                        }
                    }
                }
                _ => panic!("invalid parameter. [axis: {}, step: {}]", axis, step)
            }
        }
    }
}

pub fn slice_d3(x: &[f32], xi: usize, xj: usize, xk: usize, axis: usize, start: usize, end: usize, step: isize, result: &mut [f32]) {
    match axis {
        0 => slice_d2(x, xi, xj * xk, 0, start, end, step, result),
        1 => {
            let indices = create_indices(start, end, step);
            let size = indices.len();
            for i in 0..xi {
                let xii = i * xj * xk;
                let rii = i * size * xk;
                for (r_i, x_i) in indices.iter().enumerate() {
                    let x_offset = xii + x_i * xk;
                    let result_offset = rii + r_i * xk;
                    result[result_offset..result_offset + xk].copy_from_slice(&x[x_offset..x_offset + xk]);
                }
            }
        }
        2 => slice_d2(x, xi * xj, xk, 1, start, end, step, result),
        _ => panic!("invalid parameter. [axis: {}, step: {}]", axis, step),
    }
}

fn create_indices(start: usize, end: usize, step: isize) -> Vec<usize> {
    match step {
        step if step > 0 => {
            (start..=end).step_by(step as usize).collect()
        },
        step if step < 0 => {
            (end..=start).rev().step_by(step.unsigned_abs()).collect()
        },
        _ => panic!("invalid parameter. [step: {}]",step)
    }
}
