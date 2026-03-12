use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use gpu::ops;
use gpu::resource::buffer::GPUBuffer;
use gpu::runtime::Runtime;

fn bench_compare(c: &mut Criterion) {
    let mut runtime = ops::runtime::allocate();
    let mut group = c.benchmark_group("compare");

    bench_d1_to_d0(&mut group, "gt_d1_to_d0", &mut runtime, |_, x, y, res, rt| {
        ops::compare::gt_d1_to_d0(x, y, res, rt);
    });
    bench_d1_to_d1(&mut group, "gt_d1_to_d1", &mut runtime, |_, x, y, res, rt| {
        ops::compare::gt_d1_to_d1(x, y, res, rt);
    });

    bench_d1_to_d0(&mut group, "lt_d1_to_d0", &mut runtime, |_, x, y, res, rt| {
        ops::compare::lt_d1_to_d0(x, y, res, rt);
    });
    bench_d1_to_d1(&mut group, "lt_d1_to_d1", &mut runtime, |_, x, y, res, rt| {
        ops::compare::lt_d1_to_d1(x, y, res, rt);
    });

    bench_d1_to_d0(&mut group, "eq_d1_to_d0", &mut runtime, |_, x, y, res, rt| {
        ops::compare::eq_d1_to_d0(x, y, 0f32, 0f32, res, rt);
    });
    bench_d1_to_d1(&mut group, "eq_d1_to_d1", &mut runtime, |_, x, y, res, rt| {
        ops::compare::eq_d1_to_d1(x, y, 0f32, 0f32,res, rt);
    });

    bench_d0_to_d0(&mut group, "where_d0_to_d0", &mut runtime, |condition, x, y, res, rt| {
        ops::compare::where_d0_to_d0(condition, x, y, res, rt);
    });
    bench_d0_to_d1(&mut group, "where_d0_to_d1", &mut runtime, |condition, x, y, res, rt| {
        ops::compare::where_d0_to_d1(condition, x, y, res, rt);
    });
    bench_d1_to_d0(&mut group, "where_d1_to_d0", &mut runtime, |condition, x, y, res, rt| {
        ops::compare::where_d1_to_d0(condition, x, y, res, rt);
    });
    bench_d1_to_d1(&mut group, "where_d1_to_d1", &mut runtime, |condition, x, y, res, rt| {
        ops::compare::where_d1_to_d1(condition, x, y,res, rt);
    });

    group.finish();
}

fn bench_d0_to_d0<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, f32, f32, &GPUBuffer, &mut Runtime),
{
    for size in [1_000, 1_000_000] {
        let condition = ops::buffer::init(&vec![1.0f32; size], runtime);
        let x = 1f32;
        let y = 1f32;
        let result = ops::buffer::create(1, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("size_{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&condition, x, y, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

fn bench_d0_to_d1<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, f32, &GPUBuffer, &GPUBuffer, &mut Runtime),
{
    for size in [1_000, 1_000_000] {
        let condition = ops::buffer::init(&vec![1.0f32; size], runtime);
        let x = 1f32;
        let y = ops::buffer::init(&vec![1.0f32; size], runtime);
        let result = ops::buffer::create(1, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("size_{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&condition, x, &y, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

fn bench_d1_to_d0<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, &GPUBuffer, f32, &GPUBuffer, &mut Runtime),
{
    for size in [1_000, 1_000_000] {
        let condition = ops::buffer::init(&vec![1.0f32; size], runtime);
        let x = ops::buffer::init(&vec![1.0f32; size], runtime);
        let y = 1f32;
        let result = ops::buffer::create(1, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("size_{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&condition, &x, y, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

fn bench_d1_to_d1<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where
    F: Fn(&GPUBuffer, &GPUBuffer, &GPUBuffer, &GPUBuffer, &mut Runtime),
{
    for size in [1_000, 1_000_000] {
        let condition = ops::buffer::init(&vec![1.0f32; size], runtime);
        let x = ops::buffer::init(&vec![1.0f32; size], runtime);
        let y = ops::buffer::init(&vec![1.0f32; size], runtime);
        let result = ops::buffer::create(1, runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, format!("size_{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        op(&condition, &x, &y, &result, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
