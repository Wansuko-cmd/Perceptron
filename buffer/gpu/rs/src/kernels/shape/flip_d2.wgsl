struct Params {
    xi: u32,
    xj: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1)
fn flip_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let old_index = id.y * stride + id.x;
    if (old_index >= arrayLength(&x)) {
        return;
    }

    var tmp = old_index;
    let oj = tmp % params.xj; tmp = tmp / params.xj;
    let oi = tmp;

    let result_index = (params.xi - oi - 1) * params.xj + oj;
    result[result_index] = x[old_index];
}

@compute @workgroup_size(256, 1)
fn flip_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let old_index = id.y * stride + id.x;
    if (old_index >= arrayLength(&x)) {
        return;
    }

    var tmp = old_index;
    let oj = tmp % params.xj; tmp = tmp / params.xj;
    let oi = tmp;

    let result_index = oi * params.xj + (params.xj - oj - 1);
    result[result_index] = x[old_index];
}
