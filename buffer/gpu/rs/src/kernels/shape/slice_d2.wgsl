struct Params {
    start: u32,
    step: i32,
    size: u32,
    xi: u32,
    xj: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn slice_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }
    let ni = result_index / params.xj;
    let nj = result_index % params.xj;

    let x_offset = (i32(params.start) + i32(ni) * params.step) * i32(params.xj);
    result[result_index] = x[x_offset + i32(nj)];
}

@compute @workgroup_size(256, 1)
fn slice_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }
    let ni = result_index / params.size;
    let nj = result_index % params.size;

    let x_offset = i32(ni * params.xj);
    let x_index = x_offset + i32(params.start) + i32(nj) * params.step;
    result[result_index] = x[x_index];
}
