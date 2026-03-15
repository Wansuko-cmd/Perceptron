struct Params {
    start: u32,
    step: i32,
    count : u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1)
fn copy_into(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let old_index = id.y * stride + id.x;
    if (old_index >= params.count || old_index >= arrayLength(&x)) {
        return;
    }

    let new_index = i32(params.start) + i32(old_index) * params.step;
    if (new_index >= 0 && u32(new_index) < arrayLength(&result)) {
        result[new_index] = x[old_index];
    }
}
