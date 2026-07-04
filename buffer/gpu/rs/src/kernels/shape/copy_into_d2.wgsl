struct Params {
    start: u32,
    step: i32,
    size: u32,
    ri: u32,
    rj: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn copy_into_d2_axis0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let x_index = id.y * stride + id.x;
    if (x_index >= arrayLength(&x)) {
        return;
    }
    let oi = x_index / params.rj;
    let oj = x_index % params.rj;

    let result_offset = (i32(params.start) + i32(oi) * params.step) * i32(params.rj);
    result[result_offset + i32(oj)] = x[x_index];
}

@compute @workgroup_size(256, 1)
fn copy_into_d2_axis1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let x_index = id.y * stride + id.x;
    if (x_index >= arrayLength(&x)) {
        return;
    }
    let oi = x_index / params.size;
    let oj = x_index % params.size;

    let result_offset = i32(oi * params.rj);
    let result_index = result_offset + i32(params.start) + i32(oj) * params.step;
    result[result_index] = x[x_index];
}
