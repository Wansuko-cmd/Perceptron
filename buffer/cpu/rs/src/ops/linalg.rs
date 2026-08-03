use matrixmultiply::sgemm;
use rayon::prelude::*;

use crate::resource::buffer::CPUBuffer;

pub fn inner(x: &CPUBuffer, y: &CPUBuffer, b: usize, result: &mut CPUBuffer) {
    assert_eq!(x.count(), y.count());
    let n = x.count() / b;
    for i in 0..b {
        let x_ptr = &x[i * n..(i + 1) * n];
        let y_ptr = &y[i * n..(i + 1) * n];

        result[i] = x_ptr.iter().zip(y_ptr).map(|(a, b)| a * b).sum();
    }
}

pub fn mat_mul_d1_to_d2(
    x: &CPUBuffer,
    y: &CPUBuffer,
    trans_y: bool,
    n: usize,
    k: usize,
    result: &mut CPUBuffer,
) {
    let m = 1;

    let rsa = k as isize;
    let csa = 1;

    let (rsb, csb) = match trans_y {
        true => (1, k as isize),
        false => (n as isize, 1),
    };

    let rsc = n as isize;
    let csc = 1;

    unsafe {
        sgemm(
            m, k, n,
            1.0,
            x.as_ptr(), rsa, csa,
            y.as_ptr(), rsb, csb,
            0.0,
            result.as_mut_ptr(), rsc, csc,
        );
    }
}

pub fn mat_mul_d2_to_d1(
    x: &CPUBuffer,
    trans_x: bool,
    y: &CPUBuffer,
    m: usize,
    k: usize,
    result: &mut CPUBuffer,
) {
    let n = 1;

    let (rsa, csa) = match trans_x {
        true => (1, m as isize),
        false => (k as isize, 1)
    };

    let rsb = 1;
    let csb = 1;

    let rsc = 1;
    let csc = 1;

    unsafe {
        sgemm(
            m, k, n,
            1.0,
            x.as_ptr(), rsa, csa,
            y.as_ptr(), rsb, csb,
            0.0,
            result.as_mut_ptr(), rsc, csc,
        );
    }
}

pub fn mat_mul_d2_to_d2(
    x: &CPUBuffer,
    trans_x: bool,
    y: &CPUBuffer,
    trans_y: bool,
    m: usize,
    n: usize,
    k: usize,
    _b: usize,
    result: &mut CPUBuffer,
) {
    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;

    let (rsa, csa) = match trans_x {
        true => (1, m as isize),
        false => (k as isize, 1)
    };
    let (rsb, csb) = match trans_y {
        true => (1, k as isize),
        false => (n as isize, 1)
    };
    let rsc = n as isize;
    let csc = 1;

    result.par_chunks_mut(stride_c)
        .enumerate()
        .for_each(|(i, c_ptr)| {
            let a_ptr = &x[i * stride_a..];
            let b_ptr = &y[i * stride_b..];

            unsafe {
                sgemm(
                    m, k, n,
                    1.0,
                    a_ptr.as_ptr(), rsa, csa,
                    b_ptr.as_ptr(), rsb, csb,
                    0.0,
                    c_ptr.as_mut_ptr(), rsc, csc,
                );
            }
        });
}
