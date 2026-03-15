struct Params {
    start: u32,
    step: i32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1)
fn slice(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let new_index = id.y * stride + id.x;
    if (new_index >= arrayLength(&result)) {
        return;
    }

    let old_index = i32(params.start) + i32(new_index) * params.step;
    result[new_index] = x[old_index];
}
