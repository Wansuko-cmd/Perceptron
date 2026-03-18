pub fn exp_d1(x: &[f32], result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = val.exp();
    }
}

pub fn ln_d1(x: &[f32], e: f32, result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = (val + e).ln();
    }
}

pub fn sigmoid_d1(x: &[f32], result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = 1f32 / (1f32 + (-val).exp());
    }
}

pub fn pow_d1(x: &[f32], n: i32, result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = val.powi(n);
    }
}

pub fn sqrt_d1(x: &[f32], e: f32, result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = (val + e).sqrt();
    }
}
