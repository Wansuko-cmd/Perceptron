struct Params {
    new_stride: vec4<u32>,
    permuted_stride: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1)
fn transpose_d4(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let new_index = id.y * stride + id.x;
    if (new_index >= arrayLength(&result)) {
        return;
    }

    let ni = new_index / params.new_stride.x;
    let nj = (new_index % params.new_stride.x) / params.new_stride.y;
    let nk = (new_index % params.new_stride.y) / params.new_stride.z;
    let nl = new_index % params.new_stride.z;
    let coords = vec4<u32>(ni, nj, nk, nl);

    let old_index = dot(coords, params.permuted_stride);

    result[new_index] = x[old_index];
}
