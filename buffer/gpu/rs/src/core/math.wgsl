struct Params {
    val: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn exp_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input_buf)) {
        return;
    }

    output_buf[idx] = exp(input_buf[idx]);
}

@compute @workgroup_size(64)
fn ln_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let e = params.val;
    if (idx >= arrayLength(&input_buf)) {
        return;
    }

    output_buf[idx] = log(input_buf[idx] + e);
}

@compute @workgroup_size(64)
fn sigmoid_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input_buf)) {
        return;
    }

    output_buf[idx] = 1 / (1 + exp(-input_buf[idx]));
}

@compute @workgroup_size(64)
fn pow_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let n = params.val;
    if (idx >= arrayLength(&input_buf)) {
        return;
    }

    if (params.val % 2 == 0) {
        output_buf[idx] = pow(abs(input_buf[idx]), params.val);
    } else {
        output_buf[idx] = sign(input_buf[idx]) * pow(abs(input_buf[idx]), params.val);
    }
}

@compute @workgroup_size(64)
fn sqrt_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let e = params.val;
    if (idx >= arrayLength(&input_buf)) {
        return;
    }

    output_buf[idx] = sqrt(input_buf[idx] + e);
}
