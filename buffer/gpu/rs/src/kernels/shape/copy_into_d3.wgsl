struct Params {
    start: u32,
    step: i32,
    size: u32,
    ri: u32,
    rj: u32,
    rk: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn copy_into_d3(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let x_index = id.y * stride + id.x;
    if (x_index >= arrayLength(&x)) {
        return;
    }

    var tmp = x_index;
    let ok = tmp % params.rk; tmp = tmp / params.rk;
    let oj = tmp % params.size; tmp = tmp / params.size;
    let oi = tmp % params.ri;

    let rii = i32(oi * params.rj * params.rk);
    let result_offset = rii + (i32(params.start) + i32(oj) * params.step) * i32(params.rk);
    let result_index = result_offset + i32(ok);
    result[result_index] = x[x_index];
}
