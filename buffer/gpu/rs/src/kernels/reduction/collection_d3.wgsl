struct Params {
    xi: u32,
    xj: u32,
    xk: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn min_d3(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let k = index % params.xk;
    let i = index / params.xk;

    let start = i * (params.xj * params.xk) + k;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j * params.xk;
        acc = min(acc, x[offset]);
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn max_d3(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let k = index % params.xk;
    let i = index / params.xk;

    let start = i * (params.xj * params.xk) + k;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j * params.xk;
        acc = max(acc, x[offset]);
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn sum_d3(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let k = index % params.xk;
    let i = index / params.xk;

    let start = i * (params.xj * params.xk) + k;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j * params.xk;
        acc += x[offset];
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn average_d3(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let k = index % params.xk;
    let i = index / params.xk;

    let start = i * (params.xj * params.xk) + k;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j * params.xk;
        acc += x[offset];
    }

    result[index] = acc / f32(params.xj);
}
