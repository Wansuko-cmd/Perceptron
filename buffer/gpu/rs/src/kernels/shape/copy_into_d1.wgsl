struct Params {
    start: u32,
    step: i32,
    size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn copy_into_d1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let x_index = id.y * stride + id.x;
    if (x_index >= arrayLength(&x) || x_index >= params.size) {
        return;
    }

    let result_index = i32(params.start) + i32(x_index) * params.step;
    result[result_index] = x[x_index];
}
