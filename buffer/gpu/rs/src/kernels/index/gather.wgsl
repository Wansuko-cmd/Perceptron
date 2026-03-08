struct Params {
    i: u32,
    j: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1)
fn gather(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let n = arrayLength(&x);
    var tmp = index;
    let rk = tmp % params.k; tmp = tmp / params.k;
    let rj = tmp % n; tmp = tmp / n;
    let ri = tmp;

    let x_val = u32(x[rj]);
    let y_index = (ri * params.j + x_val) * params.k + rk;
    result[index] = y[y_index];
}
