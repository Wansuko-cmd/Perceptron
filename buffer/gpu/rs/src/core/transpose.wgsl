struct Params {
    new_stride: vec4<u32>,
    permuted_stride: vec4<u32>,
    length: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn transpose_d4(@builtin(global_invocation_id) id: vec3<u32>) {
    let new_index = id.x;
    if (new_index >= params.length) {
        return;
    }

    var current_idx = new_index;

    let ni = current_idx / params.new_stride[0];
    current_idx = current_idx % params.new_stride[0];

    let nj = current_idx / params.new_stride[1];
    current_idx = current_idx % params.new_stride[1];

    let nk = current_idx / params.new_stride[2];
    current_idx = current_idx % params.new_stride[2];

    let nl = current_idx;

    let old_index = ni * params.permuted_stride[0] + nj * params.permuted_stride[1] + nk * params.permuted_stride[2] + nl * params.permuted_stride[3];

    result[new_index] = x[old_index];
}
