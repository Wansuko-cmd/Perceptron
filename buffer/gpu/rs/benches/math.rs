use criterion::measurement::Measurement;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use gpu::ops;
use gpu::runtime::Runtime;
use gpu::resource::buffer::GPUBuffer;

fn bench_math(c: &mut Criterion) {
    let mut runtime = ops::runtime::allocate();
    let mut group = c.benchmark_group("math");

    bench_d1(&mut group, "exp_d1", &mut runtime, |x, res, rt| ops::elementwise::math::exp_d1(x, res, rt));
    bench_d1(&mut group, "ln_d1", &mut runtime, |x, res, rt| ops::elementwise::math::ln_d1(x, 0f32, res, rt));
    bench_d1(&mut group, "sigmoid_d1", &mut runtime, |x, res, rt| ops::elementwise::math::sigmoid_d1(x, res, rt));
    bench_d1(&mut group, "pow_d1", &mut runtime, |x, res, rt| ops::elementwise::math::pow_d1(x, 2, res, rt));
    bench_d1(&mut group, "sqrt_d1", &mut runtime, |x, res, rt| ops::elementwise::math::sqrt_d1(x, 0f32, res, rt));

    group.finish();
}

fn bench_d1<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where F: Fn(&GPUBuffer, &GPUBuffer, &mut Runtime),
{
        for size in [1_000, 1_000_000].iter() {
        let data = vec![1.0f32; *size];
        let x = ops::buffer::init(&data, &runtime);
        let result = ops::buffer::create(*size, &runtime);

        group.bench_with_input(BenchmarkId::new(func_name, size), size, |b, _| {
            b.iter(|| {
                for _ in 0..100 {
                    op(&x, &result, runtime);
                }
                runtime.submit();
                runtime.wait();
            });
        });
    }
}

criterion_group!(benches, bench_math);
criterion_main!(benches);
