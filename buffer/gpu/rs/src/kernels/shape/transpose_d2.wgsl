struct Params {
    oi: u32,
    oj: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn transpose_d2(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let ni = global_id.y;
    let nj = global_id.x;

    let new_index = ni * params.oi + nj;
    let old_index = nj * params.oj + ni;

    if (ni < params.oj && nj < params.oi) {
        result[new_index] = x[old_index];
    }
}
