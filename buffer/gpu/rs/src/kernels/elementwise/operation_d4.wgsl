struct Params {
    result_shape: vec4<u32>,
    x_stride: vec4<u32>,
    y_stride: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn plus_d4(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }

    let index = zip_with_d4(result_index);
    result[result_index] = x[index.x] + y[index.y];
}

@compute @workgroup_size(256, 1)
fn minus_d4(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }

    let index = zip_with_d4(result_index);
    result[result_index] = x[index.x] - y[index.y];
}

@compute @workgroup_size(256, 1)
fn times_d4(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }

    let index = zip_with_d4(result_index);
    result[result_index] = x[index.x] * y[index.y];
}

@compute @workgroup_size(256, 1)
fn div_d4(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let result_index = id.y * stride + id.x;
    if (result_index >= arrayLength(&result)) {
        return;
    }

    let index = zip_with_d4(result_index);
    result[result_index] = x[index.x] / y[index.y];
}

fn zip_with_d4(index: u32) -> vec2<u32> {
    var tmp = index;
    let l = tmp % params.result_shape[3]; tmp = tmp / params.result_shape[3];
    let k = tmp % params.result_shape[2]; tmp = tmp / params.result_shape[2];
    let j = tmp % params.result_shape[1]; tmp = tmp / params.result_shape[1];
    let i = tmp % params.result_shape[0];

    let coords = vec4<u32>(i, j, k, l);
    let x_index = dot(coords, params.x_stride);
    let y_index = dot(coords, params.y_stride);

    return vec2<u32>(x_index, y_index);
}
