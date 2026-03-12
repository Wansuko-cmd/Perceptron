use std::cmp::max;

use criterion::measurement::Measurement;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use gpu::ops;
use gpu::runtime::Runtime;
use gpu::resource::buffer::GPUBuffer;

fn bench_operation(c: &mut Criterion) {
    let mut runtime = ops::runtime::allocate();
    let mut group = c.benchmark_group("operation");

    for config in OperationConfig::d0_to_d1() {
        bench(&mut group, "plus_with_d0_to_d1", &config, &mut runtime, |_, y, res, _, rt| {
            ops::operation::plus_with_d0_to_d1(1f32, y, res, rt);
        });
        bench(&mut group, "minus_with_d0_to_d1", &config, &mut runtime, |_, y, res, _, rt| {
            ops::operation::minus_with_d0_to_d1(1f32, y, res, rt);
        });
        bench(&mut group, "times_with_d0_to_d1", &config, &mut runtime, |_, y, res, _, rt| {
            ops::operation::times_with_d0_to_d1(1f32, y, res, rt);
        });
        bench(&mut group, "div_with_d0_to_d1", &config, &mut runtime, |_, y, res, _, rt| {
            ops::operation::div_with_d0_to_d1(1f32, y, res, rt);
        });
    }

    for config in OperationConfig::d1_to_d0() {
        bench(&mut group, "plus_with_d1_to_d0", &config, &mut runtime, |x, _, res, _, rt| {
            ops::operation::plus_with_d1_to_d0(x, 1f32, res, rt);
        });
        bench(&mut group, "minus_with_d1_to_d0", &config, &mut runtime, |x, _, res, _, rt| {
            ops::operation::minus_with_d1_to_d0(x, 1f32, res, rt);
        });
        bench(&mut group, "times_with_d1_to_d0", &config, &mut runtime, |x, _, res, _, rt| {
            ops::operation::times_with_d1_to_d0(x, 1f32, res, rt);
        });
        bench(&mut group, "div_with_d1_to_d0", &config, &mut runtime, |x, _, res, _, rt| {
            ops::operation::div_with_d1_to_d0(x, 1f32, res, rt);
        });
    }

    for config in OperationConfig::d1_to_d1() {
        bench(&mut group, "plus_with_d1_to_d1", &config, &mut runtime, |x, y, res, _, rt| {
            ops::operation::plus_with_d1_to_d1(x, y, res, rt);
        });
        bench(&mut group, "minus_with_d1_to_d1", &config, &mut runtime, |x, y, res, _, rt| {
            ops::operation::minus_with_d1_to_d1(x, y, res, rt);
        });
        bench(&mut group, "times_with_d1_to_d1", &config, &mut runtime, |x, y, res, _, rt| {
            ops::operation::times_with_d1_to_d1(x, y, res, rt);
        });
        bench(&mut group, "div_with_d1_to_d1", &config, &mut runtime, |x, y, res, _, rt| {
            ops::operation::div_with_d1_to_d1(x, y, res, rt);
        });
    }

    for config in OperationConfig::d1_to_d2() {
        bench(&mut group, "plus_with_d1_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d1_to_d2(x, y, config.yi, config.yj, 0, res, rt);
        });
        bench(&mut group, "minus_with_d1_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d1_to_d2(x, y, config.yi, config.yj, 0, res, rt);
        });
        bench(&mut group, "times_with_d1_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d1_to_d2(x, y, config.yi, config.yj, 0, res, rt);
        });
        bench(&mut group, "div_with_d1_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d1_to_d2(x, y, config.yi, config.yj, 0, res, rt);
        });
    }

    for config in OperationConfig::d1_to_d3() {
        bench(&mut group, "plus_with_d1_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d1_to_d3(x, y, config.yi, config.yj, config.yk, 0, res, rt);
        });
        bench(&mut group, "minus_with_d1_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d1_to_d3(x, y, config.yi, config.yj, config.yk, 0, res, rt);
        });
        bench(&mut group, "times_with_d1_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d1_to_d3(x, y, config.yi, config.yj, config.yk, 0, res, rt);
        });
        bench(&mut group, "div_with_d1_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d1_to_d3(x, y, config.yi, config.yj, config.yk, 0, res, rt);
        });
    }

    for config in OperationConfig::d2_to_d1() {
        bench(&mut group, "plus_with_d2_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d2_to_d1(x, config.xi, config.xj, y, 0, res, rt);
        });
        bench(&mut group, "minus_with_d2_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d2_to_d1(x, config.xi, config.xj, y, 0, res, rt);
        });
        bench(&mut group, "times_with_d2_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d2_to_d1(x, config.xi, config.xj, y, 0, res, rt);
        });
        bench(&mut group, "div_with_d2_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d2_to_d1(x, config.xi, config.xj, y, 0, res, rt);
        });
    }

    for config in OperationConfig::d2_to_d3() {
        bench(&mut group, "plus_with_d2_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d2_to_d3(x, config.xi, config.xj, y, config.yi, config.yj, config.yk, 0, 1, res, rt);
        });
        bench(&mut group, "minus_with_d2_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d2_to_d3(x, config.xi, config.xj, y, config.yi, config.yj, config.yk, 0, 1, res, rt);
        });
        bench(&mut group, "times_with_d2_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d2_to_d3(x, config.xi, config.xj, y, config.yi, config.yj, config.yk, 0, 1, res, rt);
        });
        bench(&mut group, "div_with_d2_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d2_to_d3(x, config.xi, config.xj, y, config.yi, config.yj, config.yk, 0, 1, res, rt);
        });
    }

    for config in OperationConfig::d3_to_d1() {
        bench(&mut group, "plus_with_d3_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d3_to_d1(x, config.xi, config.xj, config.xk, y, 0, res, rt);
        });
        bench(&mut group, "minus_with_d3_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d3_to_d1(x, config.xi, config.xj, config.xk, y, 0, res, rt);
        });
        bench(&mut group, "times_with_d3_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d3_to_d1(x, config.xi, config.xj, config.xk, y, 0, res, rt);
        });
        bench(&mut group, "div_with_d3_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d3_to_d1(x, config.xi, config.xj, config.xk, y, 0, res, rt);
        });
    }

    for config in OperationConfig::d3_to_d2() {
        bench(&mut group, "plus_with_d3_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d3_to_d2(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, 0, 1, res, rt);
        });
        bench(&mut group, "minus_with_d3_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d3_to_d2(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, 0, 1, res, rt);
        });
        bench(&mut group, "times_with_d3_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d3_to_d2(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, 0, 1, res, rt);
        });
        bench(&mut group, "div_with_d3_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d3_to_d2(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, 0, 1, res, rt);
        });
    }

    for config in OperationConfig::d3_to_d4() {
        bench(&mut group, "plus_with_d3_to_d4", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d3_to_d4(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, config.yk, config.yl, 0, 1, 2, res, rt);
        });
        bench(&mut group, "minus_with_d3_to_d4", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d3_to_d4(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, config.yk, config.yl, 0, 1, 2, res, rt);
        });
        bench(&mut group, "times_with_d3_to_d4", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d3_to_d4(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, config.yk, config.yl, 0, 1, 2, res, rt);
        });
        bench(&mut group, "div_with_d3_to_d4", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d3_to_d4(x, config.xi, config.xj, config.xk, y, config.yi, config.yj, config.yk, config.yl, 0, 1, 2, res, rt);
        });
    }

    for config in OperationConfig::d4_to_d1() {
        bench(&mut group, "plus_with_d4_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d4_to_d1(x, config.xi, config.xj, config.xk, config.xl, y, 0, res, rt);
        });
        bench(&mut group, "minus_with_d4_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d4_to_d1(x, config.xi, config.xj, config.xk, config.xl, y, 0, res, rt);
        });
        bench(&mut group, "times_with_d4_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d4_to_d1(x, config.xi, config.xj, config.xk, config.xl, y, 0, res, rt);
        });
        bench(&mut group, "div_with_d4_to_d1", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d4_to_d1(x, config.xi, config.xj, config.xk, config.xl, y, 0, res, rt);
        });
    }

    for config in OperationConfig::d4_to_d2() {
        bench(&mut group, "plus_with_d4_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d4_to_d2(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, 0, 1, res, rt);
        });
        bench(&mut group, "minus_with_d4_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d4_to_d2(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, 0, 1, res, rt);
        });
        bench(&mut group, "times_with_d4_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d4_to_d2(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, 0, 1, res, rt);
        });
        bench(&mut group, "div_with_d4_to_d2", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d4_to_d2(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, 0, 1, res, rt);
        });
    }

    for config in OperationConfig::d4_to_d3() {
        bench(&mut group, "plus_with_d4_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::plus_with_d4_to_d3(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, config.yk, 0, 1, 2, res, rt);
        });
        bench(&mut group, "minus_with_d4_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::minus_with_d4_to_d3(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, config.yk, 0, 1, 2, res, rt);
        });
        bench(&mut group, "times_with_d4_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::times_with_d4_to_d3(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, config.yk, 0, 1, 2, res, rt);
        });
        bench(&mut group, "div_with_d4_to_d3", &config, &mut runtime, |x, y, res, config, rt| {
            ops::operation::div_with_d4_to_d3(x, config.xi, config.xj, config.xk, config.xl, y, config.yi, config.yj, config.yk, 0, 1, 2, res, rt);
        });
    }

    group.finish();
}

struct OperationConfig {
    xi: usize,
    xj: usize,
    xk: usize,
    xl: usize,
    yi: usize,
    yj: usize,
    yk: usize,
    yl: usize,
    repeat: usize,
}

impl Default for OperationConfig {
    fn default() -> Self {
        Self { xi: 1, xj: 1, xk: 1, xl: 1, yi: 1, yj: 1, yk: 1, yl: 1, repeat: 0 }
    }
}

impl OperationConfig {
    fn d0_to_d1() -> [OperationConfig; 2] {
        [
            OperationConfig { yi: 128, repeat: 100, ..Default::default() },
            OperationConfig { yi: 1024, repeat: 10, ..Default::default() },
        ]
    }

    fn d1_to_d0() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 128, repeat: 100, ..Default::default() },
            OperationConfig { xi: 1024, repeat: 10, ..Default::default() },
        ]
    }

    fn d1_to_d1() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 128, yi: 128, repeat: 100, ..Default::default() },
            OperationConfig { xi: 1024, yi: 1024, repeat: 10, ..Default::default() },
        ]
    }

    fn d1_to_d2() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 128, yi: 128, yj: 128, repeat: 100, ..Default::default() },
            OperationConfig { xi: 1024, yi: 1024, yj: 1024, repeat: 10, ..Default::default() },
        ]
    }

    fn d1_to_d3() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 16, yi: 16, yj: 16, yk: 16, repeat: 100, ..Default::default() },
            OperationConfig { xi: 128, yi: 128, yj: 128, yk: 128, repeat: 10, ..Default::default() },       
        ]
    }

    fn d2_to_d1() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 128, xj: 128, yi: 128, repeat: 100, ..Default::default() },
            OperationConfig { xi: 1024, xj: 1024, yi: 1024, repeat: 10, ..Default::default() },
        ]
    }

    fn d2_to_d3() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 16, xj: 16, yi: 16, yj: 16, yk: 16, repeat: 100, ..Default::default() },
            OperationConfig { xi: 128, xj: 128, yi: 128, yj: 128, yk: 128, repeat: 10, ..Default::default() },
        ]
    }

    fn d3_to_d1() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 16, xj: 16, xk: 16, yi: 16, repeat: 100, ..Default::default() },
            OperationConfig { xi: 128, xj: 128, xk: 128, yi: 128, repeat: 10, ..Default::default() },
        ]
    }

    fn d3_to_d2() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 8, xj: 8, xk: 8, yi: 8, yj: 8, repeat: 100, ..Default::default() },
            OperationConfig { xi: 128, xj: 128, xk: 128, yi: 128, yj: 128, repeat: 10, ..Default::default() },
        ]
    }

    fn d3_to_d4() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 8, xj: 8, xk: 8, yi: 8, yj: 8, yk: 8, yl: 8, repeat: 100, ..Default::default() },
            OperationConfig { xi: 64, xj: 64, xk: 64, yi: 64, yj: 64, yk: 64, yl: 64, repeat: 10, ..Default::default() },
        ]
    }

    fn d4_to_d1() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 8, xj: 8, xk: 8, xl: 8, yi: 8, repeat: 100, ..Default::default() },
            OperationConfig { xi: 64, xj: 64, xk: 64, xl: 64, yi: 64, repeat: 10, ..Default::default() },
        ]
    }

    fn d4_to_d2() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 8, xj: 8, xk: 8, xl: 8, yi: 8, yj: 8, repeat: 100, ..Default::default() },
            OperationConfig { xi: 64, xj: 64, xk: 64, xl: 64, yi: 64, yj: 64, repeat: 10, ..Default::default() },
        ]
    }

    fn d4_to_d3() -> [OperationConfig; 2] {
        [
            OperationConfig { xi: 8, xj: 8, xk: 8, xl: 8, yi: 8, yj: 8, yk: 8, repeat: 100, ..Default::default() },
            OperationConfig { xi: 64, xj: 64, xk: 64, xl: 64, yi: 64, yj: 64, yk: 64, repeat: 10, ..Default::default() },
        ]
    }
}

impl OperationConfig {
    fn name(&self) -> String {
        format!("({},{},{},{}),({},{},{},{})", self.xi, self.xj, self.xk, self.xl, self.yi, self.yj, self.yk, self.yl)
    }
}

fn bench<M: Measurement, F>(group: &mut BenchmarkGroup<M>, func_name: &str, config: &OperationConfig, runtime: &mut Runtime, op: F)
where F: Fn(&GPUBuffer, &GPUBuffer, &GPUBuffer, &OperationConfig, &mut Runtime),
{
    let OperationConfig { xi, xj, xk, xl, yi, yj, yk, yl, repeat } = config;
    let x_size = xi * xj * xk * xl;
    let y_size = yi * yj * yk * yl;
    let result_size = max(x_size, y_size);

    let x = ops::buffer::init(&vec![1.0f32; x_size], &runtime);
    let y = ops::buffer::init(&vec![1.0f32; y_size], &runtime);
    let result = ops::buffer::create(result_size, &runtime);

    group.bench_with_input(BenchmarkId::new(func_name, config.name()), &config, |b, _| {
        b.iter(|| {
            for _ in 0..*repeat {
                op(&x, &y, &result, &config, runtime);
            }
            runtime.submit();
            runtime.wait();
        });
    });
}

criterion_group!(benches, bench_operation);
criterion_main!(benches);
