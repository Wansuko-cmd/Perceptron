struct Params {
    e: f32,
    n: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn exp_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&x)) {
        return;
    }

    result[index] = exp(x[index]);
}

@compute @workgroup_size(64)
fn ln_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let e = params.e;
    if (index >= arrayLength(&x)) {
        return;
    }

    result[index] = log(x[index] + e);
}

@compute @workgroup_size(64)
fn sigmoid_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&x)) {
        return;
    }

    result[index] = 1 / (1 + exp(-x[index]));
}

@compute @workgroup_size(64)
fn pow_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let n = params.n;
    if (index >= arrayLength(&x)) {
        return;
    }

    if (params.n % 2 == 0) {
        result[index] = pow(abs(x[index]), n);
    } else {
        result[index] = sign(x[index]) * pow(abs(x[index]), n);
    }
}

@compute @workgroup_size(64)
fn sqrt_d1(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let e = params.e;
    if (index >= arrayLength(&x)) {
        return;
    }

    result[index] = sqrt(x[index] + e);
}
