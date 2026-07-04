struct Params {
    xi: u32,
    xj: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn min_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index;
    var acc = x[start];
    for (var i = 1u; i < params.xi; i++) {
        let offset = start + i * params.xj;
        acc = min(acc, x[offset]);
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn min_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index * params.xj;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j;
        acc = min(acc, x[offset]);
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn max_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index;
    var acc = x[start];
    for (var i = 1u; i < params.xi; i++) {
        let offset = start + i * params.xj;
        acc = max(acc, x[offset]);
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn max_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index * params.xj;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j;
        acc = max(acc, x[offset]);
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn sum_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index;
    var acc = x[start];
    for (var i = 1u; i < params.xi; i++) {
        let offset = start + i * params.xj;
        acc += x[offset];
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn sum_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index * params.xj;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j;
        acc += x[offset];
    }
    result[index] = acc;
}

@compute @workgroup_size(256, 1)
fn average_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index;
    var acc = x[start];
    for (var i = 1u; i < params.xi; i++) {
        let offset = start + i * params.xj;
        acc += x[offset];
    }
    result[index] = acc / f32(params.xi);
}

@compute @workgroup_size(256, 1)
fn average_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let start = index * params.xj;
    var acc = x[start];
    for (var j = 1u; j < params.xj; j++) {
        let offset = start + j;
        acc += x[offset];
    }
    result[index] = acc / f32(params.xj);
}
