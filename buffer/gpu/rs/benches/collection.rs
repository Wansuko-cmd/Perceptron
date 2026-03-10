use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use gpu::ops;
use gpu::resource::buffer::GPUBuffer;
use gpu::runtime::Runtime;

fn bench_collection(c: &mut Criterion) {
    let mut runtime = ops::runtime::allocate();
    let mut group = c.benchmark_group("collection");

    bench_d1(&mut group, "average_d1", &mut runtime, |x, res, rt| {
        ops::collection::average_d1(x, res, rt)
    });
    bench_d2(&mut group, "average_d2", &mut runtime, |x, i, j, a, res, rt| {
        ops::collection::average_d2(x, i, j, a, res, rt)
    });
    bench_d3(&mut group, "average_d3", &mut runtime, |x, i, j, k, a, res, rt| {
        ops::collection::average_d3(x, i, j, k, a, res, rt)
    });

    bench_d1(&mut group, "max_d1", &mut runtime, |x, res, rt| {
        ops::collection::max_d1(x, res, rt)
    });
    bench_d2(&mut group, "max_d2", &mut runtime, |x, i, j, a, res, rt| {
        ops::collection::max_d2(x, i, j, a, res, rt)
    });
    bench_d3(&mut group, "max_d3", &mut runtime, |x, i, j, k, a, res, rt| {
        ops::collection::max_d3(x, i, j, k, a, res, rt)
    });

    bench_d1(&mut group, "min_d1", &mut runtime, |x, res, rt| {
        ops::collection::min_d1(x, res, rt)
    });
    bench_d2(&mut group, "min_d2", &mut runtime, |x, i, j, a, res, rt| {
        ops::collection::min_d2(x, i, j, a, res, rt)
    });
    bench_d3(&mut group, "min_d3", &mut runtime, |x, i, j, k, a, res, rt| {
        ops::collection::min_d3(x, i, j, k, a, res, rt)
    });

    bench_d1(&mut group, "sum_d1", &mut runtime, |x, res, rt| {
        ops::collection::sum_d1(x, res, rt)
    });
    bench_d2(&mut group, "sum_d2", &mut runtime, |x, i, j, a, res, rt| {
        ops::collection::sum_d2(x, i, j, a, res, rt)
    });
    bench_d3(&mut group, "sum_d3", &mut runtime, |x, i, j, k, a, res, rt| {
        ops::collection::sum_d3(x, i, j, k, a, res, rt)
    });

    group.finish();
}

fn bench_d1<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, &GPUBuffer, &mut Runtime),
{
    for size in [1_000, 1_000_000] {
        let x = ops::buffer::init(&vec![1.0f32; size], runtime);
        let result = ops::buffer::create(1, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("size_{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&x, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

fn bench_d2<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, usize, usize, usize, &GPUBuffer, &mut Runtime),
{
    let axis = 0;
    for (xi, xj) in [(128, 128), (1024, 1024)] {
        let size = xi * xj;
        let x = ops::buffer::init(&vec![1.0f32; size], runtime);
        let result = ops::buffer::create(xj, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("{}x{}", xi, xj)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&x, xi, xj, axis, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

fn bench_d3<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, usize, usize, usize, usize, &GPUBuffer, &mut Runtime),
{
    let axis = 0;
    for (xi, xj, xk) in [(32, 32, 32), (128, 128, 128)] {
        let size = xi * xj * xk;
        let x = ops::buffer::init(&vec![1.0f32; size], runtime);
        let result = ops::buffer::create(xj * xk, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("{}x{}x{}", xi, xj, xk)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&x, xi, xj, xk, axis, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

criterion_group!(benches, bench_collection);
criterion_main!(benches);
