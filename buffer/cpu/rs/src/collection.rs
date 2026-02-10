pub fn average_d1(x: &[f32]) -> f32 {
    return sum_d1(x) / x.len() as f32;
}

pub fn average_d2(
    x: &[f32],
    xi: usize, xj: usize,
    axis: usize,
    result: &mut[f32],
) {
    sum_d2(x, xi, xj, axis, result);

    let n = match axis {
        0 => xi,
        1 => xj,
        _ => panic!("invalid parameter. [axis: {}]", axis),
    };

    let inv = 1f32 / n as f32;
    for res in result.iter_mut() {
        *res *= inv;
    }
}

pub fn average_d3(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &mut[f32],
) {
    sum_d3(x, xi, xj, xk, axis, result);

    let n = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        _ => panic!("invalid parameter. [axis: {}]", axis),
    };

    let inv = 1f32 / n as f32;
    for res in result.iter_mut() {
        *res *= inv;
    }
}

pub fn average_d4(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis: usize,
    result: &mut[f32],
) {
    sum_d4(x, xi, xj, xk, xl, axis, result);

    let n = match axis {
        0 => xi,
        1 => xj,
        2 => xk,
        3 => xl,
        _ => panic!("invalid parameter. [axis: {}]", axis),
    };

    let inv = 1f32 / n as f32;
    for res in result.iter_mut() {
        *res *= inv;
    }
}

pub fn max_d1(x: &[f32]) -> f32 {
    return x.iter().fold(f32::MIN, |acc, i| acc.max(*i));
}

pub fn max_d2(
    x: &[f32],
    xi: usize, xj: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(f32::MIN);
    reduce_d2(x, xi, xj, axis, result, |acc, i| acc.max(i));
}

pub fn max_d3(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(f32::MIN);
    reduce_d3(x, xi, xj, xk, axis, result, |acc, i| acc.max(i));
}

pub fn max_d4(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(f32::MIN);
    reduce_d4(x, xi, xj, xk, xl, axis, result, |acc, i| acc.max(i));
}

pub fn min_d1(x: &[f32]) -> f32 {
    return x.iter().fold(f32::MAX, |acc, i| acc.min(*i));
}

pub fn min_d2(
    x: &[f32],
    xi: usize, xj: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(f32::MAX);
    reduce_d2(x, xi, xj, axis, result, |acc, i| acc.min(i));
}

pub fn min_d3(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(f32::MAX);
    reduce_d3(x, xi, xj, xk, axis, result, |acc, i| acc.min(i));
}

pub fn min_d4(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(f32::MAX);
    reduce_d4(x, xi, xj, xk, xl, axis, result, |acc, i| acc.min(i));
}

pub fn sum_d1(x: &[f32]) -> f32 {
    return x.iter().sum();
}

pub fn sum_d2(
    x: &[f32],
    xi: usize, xj: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(0f32);
    reduce_d2(x, xi, xj, axis, result, |acc, i| acc + i);
}

pub fn sum_d3(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(0f32);
    reduce_d3(x, xi, xj, xk, axis, result, |acc, i| acc + i);
}

pub fn sum_d4(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis: usize,
    result: &mut[f32],
) {
    result.fill(0f32);
    reduce_d4(x, xi, xj, xk, xl, axis, result, |acc, i| acc + i);
}

fn reduce_d2<F: Fn(f32, f32) -> f32>(
    x: &[f32],
    xi: usize, xj: usize,
    axis: usize,
    result: &mut[f32],
    block: F,
) {
    assert!(x.len() == xi * xj);
    match axis {
        0 => {
            assert_eq!(result.len(), xj);
            for inner in x.chunks_exact(xj) {
                for (res, &val) in result.iter_mut().zip(inner) {
                    *res = block(*res, val);
                }
            }
        }
        1 => {
            assert_eq!(result.len(), xi);
            for (res, outer) in result.iter_mut().zip(x.chunks_exact(xj)) {
                *res = outer.iter().copied()
                    .fold(*res, |acc, i| block(acc, i));
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
    }
}

fn reduce_d3<F: Fn(f32, f32) -> f32>(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &mut[f32],
    block: F,
) {
    assert!(x.len() == xi * xj * xk);
    match axis {
        0 => {
            assert_eq!(result.len(), xj * xk);
            for reduction in x.chunks_exact(xj * xk) {
                 for (res, &val) in result.iter_mut().zip(reduction) {
                    *res = block(*res, val);
                }
            }
        }
        1 => {
            assert_eq!(result.len(), xi * xk);
            for (res_slice, outer) in result.chunks_exact_mut(xk).zip(x.chunks_exact(xj * xk)) {
                for reduction in outer.chunks_exact(xk) {
                    for (res, &val) in res_slice.iter_mut().zip(reduction) {
                        *res = block(*res, val);
                    }
                }
            }
        }
        2 => {
            assert_eq!(result.len(), xi * xj);
            for (res, outer) in result.iter_mut().zip(x.chunks_exact(xk)) {
                *res = outer.iter().copied()
                    .fold(*res, |acc, i| block(acc, i));
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
    }
}

fn reduce_d4<F: Fn(f32, f32) -> f32>(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis: usize,
    result: &mut[f32],
    block: F,
) {
    assert!(x.len() == xi * xj * xk * xl);
    match axis {
        0 => {
            assert_eq!(result.len(), xj * xk * xl);
            for reduction in x.chunks_exact(xj * xk * xl) {
                for (res, &val) in result.iter_mut().zip(reduction) {
                    *res = block(*res, val);
                }
            }
        }
        1 => {
            assert_eq!(result.len(), xi * xk * xl);
            for (res_slice, outer) in result.chunks_exact_mut(xk * xl).zip(x.chunks_exact(xj * xk * xl)) {
                for reduction in outer.chunks_exact(xk * xl) {
                    for (res, &val) in res_slice.iter_mut().zip(reduction) {
                        *res = block(*res, val);
                    }
                }
            }
        }
        2 => {
            assert_eq!(result.len(), xi * xj * xl);
            for (res_slice, outer) in result.chunks_exact_mut(xl).zip(x.chunks_exact(xk * xl)) {
                for reduction in outer.chunks_exact(xl) {
                    for (res, &val) in res_slice.iter_mut().zip(reduction) {
                        *res = block(*res, val);
                    }
                }
            }
        }
        3 => {
            for (res, outer) in result.iter_mut().zip(x.chunks_exact(xl)) {
                *res = outer.iter().copied()
                    .fold(*res, |acc, i| block(acc, i));
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
    }
}
