struct Params {
    m: u32,
    n: u32,
    k: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn mat_mul_nn(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let size = params.m * params.n;
    let batch_index = index / size;
    let local_index = index % size;

    let row = local_index / params.n;
    let col = local_index % params.n;

    let x_offset = batch_index * params.m * params.k;
    let y_offset = batch_index * params.k * params.n;

    var acc = 0f;
    for (var i = 0u; i < params.k; i++) {
        let x_index = x_offset + (row * params.k + i);
        let y_index = y_offset + (i * params.n + col);
        acc += x[x_index] * y[y_index];
    }
    result[index] = acc;
}

@compute @workgroup_size(256)
fn mat_mul_nt(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let size = params.m * params.n;
    let batch_index = index / size;
    let local_index = index % size;

    let row = local_index / params.n;
    let col = local_index % params.n;

    let x_offset = batch_index * params.m * params.k;
    let y_offset = batch_index * params.n * params.k;

    var acc = 0f;
    for (var i = 0u; i < params.k; i++) {
        let x_index = x_offset + (row * params.k + i);
        let y_index = y_offset + (col * params.k + i);
        acc += x[x_index] * y[y_index];
    }
    result[index] = acc;
}

@compute @workgroup_size(256)
fn mat_mul_tn(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let size = params.m * params.n;
    let batch_index = index / size;
    let local_index = index % size;

    let row = local_index / params.n;
    let col = local_index % params.n;

    let x_offset = batch_index * params.k * params.m;
    let y_offset = batch_index * params.k * params.n;

    var acc = 0f;
    for (var i = 0u; i < params.k; i++) {
        let x_index = x_offset + (i * params.m + row);
        let y_index = y_offset + (i * params.n + col);
        acc += x[x_index] * y[y_index];
    }
    result[index] = acc;
}

@compute @workgroup_size(256)
fn mat_mul_tt(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let size = params.m * params.n;
    let batch_index = index / size;
    let local_index = index % size;

    let row = local_index / params.n;
    let col = local_index % params.n;

    let x_offset = batch_index * params.k * params.m;
    let y_offset = batch_index * params.n * params.k;

    var acc = 0f;
    for (var i = 0u; i < params.k; i++) {
        let x_index = x_offset + (i * params.m + row);
        let y_index = y_offset + (col * params.k + i);
        acc += x[x_index] * y[y_index];
    }
    result[index] = acc;
}
