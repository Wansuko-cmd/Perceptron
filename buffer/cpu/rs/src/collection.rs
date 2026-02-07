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
    match axis {
        0 => {
            let inv = 1f32 / xi as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        1 => {
            let inv = 1f32 / xj as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
    }
}

pub fn average_d3(
    x: &[f32],
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &mut[f32],
) {
    sum_d3(x, xi, xj, xk, axis, result);
    match axis {
        0 => {
            let inv = 1f32 / xi as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        1 => {
            let inv = 1f32 / xj as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        2 => {
            let inv = 1f32 / xk as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
    }
}

pub fn average_d4(
    x: &[f32],
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis: usize,
    result: &mut[f32],
) {
    sum_d4(x, xi, xj, xk, xl, axis, result);
    match axis {
        0 => {
            let inv = 1f32 / xi as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        1 => {
            let inv = 1f32 / xj as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        2 => {
            let inv = 1f32 / xk as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        3 => {
            let inv = 1f32 / xl as f32;
            for res in result.iter_mut() {
                *res *= inv;
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
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
            for i in 0..xi {
                let x_i = i * xj;
                for j in 0..xj {
                    result[j] = block(result[j], x[x_i + j]);
                }
            }
        }
        1 => {
            assert_eq!(result.len(), xi);
            for i in 0..xi {
                let x_i = i * xj;
                let mut acc = result[i];
                for j in 0..xj {
                    acc = block(acc, x[x_i + j]);
                }
                result[i] = acc;
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
            for i in 0..xi {
                let x_i = i * xj;
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    let r_i = j * xk; 
                    for k in 0..xk {
                        result[r_i + k] = block(result[r_i + k], x[x_j + k]);
                    }
                }
            }
        }
        1 => {
            assert_eq!(result.len(), xi * xk);
            for i in 0..xi {
                let x_i = i * xj;
                let r_i = i * xk; 
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    for k in 0..xk {
                        result[r_i + k] = block(result[r_i + k], x[x_j + k]);
                    }
                }
            }
        }
        2 => {
            assert_eq!(result.len(), xi * xj);
            for i in 0..xi {
                let x_i = i * xj;
                let r_i = i * xj; 
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    let mut acc = result[r_i + j];
                    for k in 0..xk {
                        acc = block(acc, x[x_j + k]);
                    }
                    result[r_i + j] = acc;
                }
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
            for i in 0..xi {
                let x_i = i * xj;
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    let r_i = j * xk; 
                    for k in 0..xk {
                        let x_k = (x_j + k) * xl;
                        let r_j = (r_i + k) * xl;
                        for l in 0..xl {
                            result[r_j + l] = block(result[r_j + l], x[x_k + l]);
                        }
                    }
                }
            }
        }
        1 => {
            assert_eq!(result.len(), xi * xk * xl);
            for i in 0..xi {
                let x_i = i * xj;
                let r_i = i * xk; 
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    for k in 0..xk {
                        let x_k = (x_j + k) * xl;
                        let r_j = (r_i + k) * xl;
                        for l in 0..xl {
                            result[r_j + l] = block(result[r_j + l], x[x_k + l]);
                        }
                    }
                }
            }
        }
        2 => {
            assert_eq!(result.len(), xi * xj * xl);
            for i in 0..xi {
                let x_i = i * xj;
                let r_i = i * xj; 
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    let r_j = (r_i + j) * xl;
                    for k in 0..xk {
                        let x_k = (x_j + k) * xl;
                        for l in 0..xl {
                            result[r_j + l] = block(result[r_j + l], x[x_k + l]);
                        }
                    }
                }
            }
        }
        3 => {
            assert_eq!(result.len(), xi * xj * xk);
            for i in 0..xi {
                let x_i = i * xj;
                let r_i = i * xj; 
                for j in 0..xj {
                    let x_j = (x_i + j) * xk;
                    let r_j = (r_i + j) * xk;
                    for k in 0..xk {
                        let x_k = (x_j + k) * xl;
                        let mut acc = result[r_j + k];
                        for l in 0..xl {
                            acc = block(acc, x[x_k + l]);
                        }
                        result[r_j + k] = acc;
                    }
                }
            }
        }
        _ => panic!("invalid parameter. [axis: {}]", axis)
    }
}
