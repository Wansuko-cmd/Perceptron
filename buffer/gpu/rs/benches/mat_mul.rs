use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::Measurement};
use gpu::{ops, resource::buffer::GPUBuffer, runtime::Runtime};

fn bench_mat_mul(c: &mut Criterion) {
    let mut runtime = ops::runtime::allocate();
    let mut group = c.benchmark_group("mat_mul");

    bench(&mut group, "nn", &mut runtime, |x, y, res, config, rt| {
        ops::linalg::mat_mul::mat_mul(x, false, y, false, config.m, config.n, config.k, config.b, res, rt)
    });
    bench(&mut group, "nt", &mut runtime, |x, y, res, config, rt| {
        ops::linalg::mat_mul::mat_mul(x, false, y, true, config.m, config.n, config.k, config.b, res, rt)
    });
    bench(&mut group, "tn", &mut runtime, |x, y, res, config, rt| {
        ops::linalg::mat_mul::mat_mul(x, true, y, false, config.m, config.n, config.k, config.b, res, rt)
    });
    bench(&mut group, "tt", &mut runtime, |x, y, res, config, rt| {
        ops::linalg::mat_mul::mat_mul(x, true, y, true, config.m, config.n, config.k, config.b, res, rt)
    });

    group.finish();
}

struct MatMulConfig {
    m: usize,
    n: usize,
    k: usize,
    b: usize,
    repeat: usize,
}

impl MatMulConfig {
    fn name(&self) -> String {
        format!("m{}_n{}_k{}_b{}", self.m, self.n, self.k, self.b)
    }
}

fn bench<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, runtime: &mut Runtime, op: F)
where F: Fn(&GPUBuffer, &GPUBuffer, &GPUBuffer, &MatMulConfig, &mut Runtime)
{
    let configs = [
        MatMulConfig { m: 128, n: 128, k: 128, b: 4, repeat: 50 },
        MatMulConfig { m: 512, n: 512, k: 512, b: 4, repeat: 20 },
    ];

    for config in configs {
        let m = config.m;
        let n = config.n;
        let k = config.k;
        let b = config.b;
        let repeat = config.repeat;

        let x_size = b * m * k;
        let y_size = b * k * n;
        let res_size = b * m * n;

        let data_x = vec![1.0f32; x_size];
        let data_y = vec![1.0f32; y_size];
        let x = ops::buffer::init(&data_x, &runtime);
        let y = ops::buffer::init(&data_y, &runtime);
        let result = ops::buffer::create(res_size, &runtime);

        group.bench_with_input(
            BenchmarkId::new(func_name, config.name()),
            &config,
            |bencher, _| {
                bencher.iter(|| {
                    for _ in 0..repeat {
                        op(&x, &y, &result, &config, runtime);
                    }
                    runtime.submit();
                    runtime.wait();
                });
            },
        );
    }
}

criterion_group!(benches, bench_mat_mul);
criterion_main!(benches);
