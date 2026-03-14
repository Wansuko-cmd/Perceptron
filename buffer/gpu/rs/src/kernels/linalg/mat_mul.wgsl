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

var<workgroup> tile_x: array<f32, 256>;
var<workgroup> tile_y: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn mat_mul_nn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = local_id.y * 16u + local_id.x;

    let x_offset = batch * params.m * params.k;
    let y_offset = batch * params.k * params.n;

    var acc = 0f;
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_index = x_offset + row * params.k + xi;
        if (row < params.m && xi < params.k) {
            tile_x[t_index] = x[x_index];
        } else {
            tile_x[t_index] = 0f;
        }

        let yi = step + ty;
        let y_index = y_offset + yi * params.n + col;
        if (yi < params.k && col < params.n) {
            tile_y[t_index] = y[y_index];
        } else {
            tile_y[t_index] = 0f;
        }
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            acc += tile_x[ty * 16u + t] * tile_y[t * 16u + tx];
        }
        workgroupBarrier();
    }

    let index = (batch * params.m + row) * params.n + col;
    if (batch < params.b && row < params.m && col < params.n) {
        result[index] = acc;
    }
}

@compute @workgroup_size(16, 16, 1)
fn mat_mul_nt(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = local_id.y * 16u + local_id.x;

    let x_offset = batch * params.m * params.k;
    let y_offset = batch * params.n * params.k;

    var acc = 0f;
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_index = x_offset + row * params.k + xi;
        if (row < params.m && xi < params.k) {
            tile_x[t_index] = x[x_index];
        } else {
            tile_x[t_index] = 0f;
        }

        let yi = step + ty;
        let y_index = y_offset + col * params.k + yi;
        if (yi < params.k && col < params.n) {
            tile_y[t_index] = y[y_index];
        } else {
            tile_y[t_index] = 0f;
        }
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            acc += tile_x[ty * 16u + t] * tile_y[t * 16u + tx];
        }
        workgroupBarrier();
    }

    let index = (batch * params.m + row) * params.n + col;
    if (batch < params.b && row < params.m && col < params.n) {
        result[index] = acc;
    }
}

@compute @workgroup_size(16, 16, 1)
fn mat_mul_tn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = local_id.y * 16u + local_id.x;

    let x_offset = batch * params.k * params.m;
    let y_offset = batch * params.k * params.n;

    var acc = 0f;
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_index = x_offset + xi * params.m + row;
        if (row < params.m && xi < params.k) {
            tile_x[t_index] = x[x_index];
        } else {
            tile_x[t_index] = 0f;
        }

        let yi = step + ty;
        let y_index = y_offset + yi * params.n + col;
        if (yi < params.k && col < params.n) {
            tile_y[t_index] = y[y_index];
        } else {
            tile_y[t_index] = 0f;
        }
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            acc += tile_x[ty * 16u + t] * tile_y[t * 16u + tx];
        }
        workgroupBarrier();
    }

    let index = (batch * params.m + row) * params.n + col;
    if (batch < params.b && row < params.m && col < params.n) {
        result[index] = acc;
    }
}

@compute @workgroup_size(16, 16, 1)
fn mat_mul_tt(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = local_id.y * 16u + local_id.x;

    let x_offset = batch * params.k * params.m;
    let y_offset = batch * params.n * params.k;

    var acc = 0f;
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_index = x_offset + xi * params.m + row;
        if (row < params.m && xi < params.k) {
            tile_x[t_index] = x[x_index];
        } else {
            tile_x[t_index] = 0f;
        }

        let yi = step + ty;
        let y_index = y_offset + col * params.k + yi;
        if (yi < params.k && col < params.n) {
            tile_y[t_index] = y[y_index];
        } else {
            tile_y[t_index] = 0f;
        }
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            acc += tile_x[ty * 16u + t] * tile_y[t * 16u + tx];
        }
        workgroupBarrier();
    }

    let index = (batch * params.m + row) * params.n + col;
    if (batch < params.b && row < params.m && col < params.n) {
        result[index] = acc;
    }
}
