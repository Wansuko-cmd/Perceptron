use criterion::measurement::Measurement;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use gpu::ops;
use gpu::resource::buffer::GPUBuffer;
use gpu::runtime::Runtime;

fn bench_transpose(c: &mut Criterion) {
    let mut runtime = ops::runtime::allocate();
    let mut group = c.benchmark_group("transpose");

    for (xi, xj) in [(32, 32), (1000, 1000)] {
        bench(&mut group, "d2", xi * xj, &mut runtime, |x, res, rt| {
            ops::transpose::transpose_d2(x, xi, xj, res, rt)
        });
    }

    for (xi, xj, xk) in [(10, 10, 10), (100, 100, 100)] {
        bench(&mut group, "d3", xi * xj, &mut runtime, |x, res, rt| {
            ops::transpose::transpose_d3(x, xi, xj, xk, 2, 1, 0, res, rt)
        });
    }

    for (xi, xj, xk, xl) in [(4, 4, 8, 8), (32, 32, 32, 32)] {
        bench(&mut group, "d4", xi * xj, &mut runtime, |x, res, rt| {
            ops::transpose::transpose_d4(x, xi, xj, xk, xl, 3, 2, 1, 0, res, rt)
        });
    }

    group.finish();
}

fn bench<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, size: usize, runtime: &mut Runtime, op: F)
where F: Fn(&GPUBuffer, &GPUBuffer, &mut Runtime)
{
    let data = vec![1.0f32; size];
    let x = ops::buffer::init(&data, &runtime);
    let result = ops::buffer::create(size, &runtime);

    group.bench_with_input(
        BenchmarkId::new(func_name, size),
        &size,
        |b, _| {
            b.iter(|| {
                for _ in 0..100 {
                    op(&x, &result, runtime);
                }
                runtime.submit();
                runtime.wait();
            });
        },
    );
}

criterion_group!(benches, bench_transpose);
criterion_main!(benches);
