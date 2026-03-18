pub fn gather(x: &[f32], y: &[f32], i: usize, j: usize, k: usize, result: &mut [f32]) {
    let n = x.len();
    assert_eq!(y.len(), i * j * k);
    assert_eq!(result.len(), i * n * k);

    for (y_i, result_i) in y.chunks_exact(j * k).zip(result.chunks_exact_mut(n * k)) {
        for (rj, res) in result_i.chunks_exact_mut(k).enumerate() {
            let index = x[rj] as usize;
            let y_start = index * k;
            res.copy_from_slice(&y_i[y_start..y_start + k]);
        }
    }
}

pub fn scatter_add(x: &[f32], y: &[f32], i: usize, j: usize, k: usize, b: usize, result: &mut [f32]) {
    let n = y.len() / b;
    assert_eq!(x.len(), b * i * n * k);
    assert_eq!(result.len(), i * j * k);

    for xb in 0..b {
        for xi in 0..i {
            for xj in 0..n {
                let index = y[xj] as usize;
                let x_offset = ((xb * i + xi) * n + xj) * k;
                let result_offset = (xi * j + index) * k;
                
                let x_view = x[x_offset..x_offset + k].iter();
                let result_view = result[result_offset..result_offset + k].iter_mut();
                for (res, val) in result_view.zip(x_view) {
                    *res += val;
                }
            }
        }
    }
}
