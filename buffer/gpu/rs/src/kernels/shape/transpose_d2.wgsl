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
fn transpose_d2(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }

    let ni = result_index / params.xi;
    let nj = result_index % params.xi;

    let x_index = nj * params.xj + ni;
    result[result_index] = x[x_index];
}
