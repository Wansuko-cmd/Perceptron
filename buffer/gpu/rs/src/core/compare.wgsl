struct Params {
    x_val: f32,
    y_val: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> condition: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn gt_d1_to_d0(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = select(0f, 1f, x[index] > params.y_val);
}

@compute @workgroup_size(256)
fn gt_d1_to_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = select(0f, 1f, x[index] > y[index]);
}

@compute @workgroup_size(256)
fn lt_d1_to_d0(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = select(0f, 1f, x[index] < params.y_val);
}

@compute @workgroup_size(256)
fn lt_d1_to_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = select(0f, 1f, x[index] < y[index]);
}

@compute @workgroup_size(256)
fn where_d0_to_d0(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
   result[index] = select(params.y_val, params.x_val, condition[index] > 0.0);
}

@compute @workgroup_size(256)
fn where_d0_to_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
   result[index] = select(y[index], params.x_val, condition[index] > 0.0);
}

@compute @workgroup_size(256)
fn where_d1_to_d0(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = select(params.y_val, x[index], condition[index] > 0.0);
}

@compute @workgroup_size(256)
fn where_d1_to_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = select(y[index], x[index], condition[index] > 0.0);
}
