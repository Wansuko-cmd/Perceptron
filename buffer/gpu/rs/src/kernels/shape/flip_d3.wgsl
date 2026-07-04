struct Params {
    xj: u32,
    xk: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn flip_d3(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let x_index = id.y * stride + id.x;
    if (x_index >= arrayLength(&x)) {
        return;
    }

    var tmp = x_index;
    let ok = tmp % params.xk; tmp = tmp / params.xk;
    let oj = tmp % params.xj; tmp = tmp / params.xj;
    let oi = tmp;

    let result_index = (oi * params.xj + (params.xj - oj - 1)) * params.xk + ok;
    result[result_index] = x[x_index];
}
