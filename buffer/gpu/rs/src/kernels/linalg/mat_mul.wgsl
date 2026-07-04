struct Params {
    m: u32,
    n: u32,
    k: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

var<workgroup> tile_x: array<vec4<f32>, 256>;
var<workgroup> tile_y: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn mat_mul_nn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row_vec_index = global_id.y * 4u;
    let col_vec_index = global_id.x * 4u;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = ty * 16u + tx;

    let xb_offset = batch * params.m * params.k;
    let yb_offset = batch * params.k * params.n;

    var acc0 = vec4<f32>(0f);
    var acc1 = vec4<f32>(0f);
    var acc2 = vec4<f32>(0f);
    var acc3 = vec4<f32>(0f);
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_offset = xb_offset + row_vec_index * params.k + xi;
        var x_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (row_vec_index + i < params.m && xi < params.k) {
                let x_index = x_offset + i * params.k;
                x_tmp[i] = x[x_index];
            }
        }
        tile_x[t_index] = x_tmp;

        let yi = step + ty;
        let y_offset = yb_offset + yi * params.n + col_vec_index;
        var y_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (yi < params.k && col_vec_index + i < params.n) {
                let y_index = y_offset + i;
                y_tmp[i] = y[y_index];
            }
        }
        tile_y[t_index] = y_tmp;
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            let x_val = tile_x[ty * 16u + t];
            let y_val = tile_y[t * 16u + tx];
            acc0 = fma(vec4<f32>(x_val[0u]), y_val, acc0);
            acc1 = fma(vec4<f32>(x_val[1u]), y_val, acc1);
            acc2 = fma(vec4<f32>(x_val[2u]), y_val, acc2);
            acc3 = fma(vec4<f32>(x_val[3u]), y_val, acc3);
        }
        workgroupBarrier();
    }

    let rb_offset = batch * params.m * params.n;
    let acc = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);

    for (var ri = 0u; ri < 4u; ri++) {
        let row_index = row_vec_index + ri;
        if (batch < params.b && row_index < params.m) {
            let offset = rb_offset + row_index * params.n;
            for (var ci = 0u; ci < 4u; ci++) {
                let col_index = col_vec_index + ci;
                if (col_index < params.n) {
                    result[offset + col_index] = acc[ri][ci];
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn mat_mul_nt(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row_vec_index = global_id.y * 4u;
    let col_vec_index = global_id.x * 4u;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = ty * 16u + tx;

    let xb_offset = batch * params.m * params.k;
    let yb_offset = batch * params.n * params.k;

    // (64, 16)の領域をコピー
    // 縦方向に4つずつ取っていくため、Y軸方向の割当は64/4=16
    let yt_x = tx;
    let yt_y_vec = ty * 4u;
    let yt_row_offset = workgroup_id.x * 64u + yt_y_vec;

    var acc0 = vec4<f32>(0f);
    var acc1 = vec4<f32>(0f);
    var acc2 = vec4<f32>(0f);
    var acc3 = vec4<f32>(0f);
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_offset = xb_offset + row_vec_index * params.k + xi;
        var x_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (row_vec_index + i < params.m && xi < params.k) {
                let x_index = x_offset + i * params.k;
                x_tmp[i] = x[x_index];
            }
        }
        tile_x[t_index] = x_tmp;

        let yt_col_index = step + yt_x;
        let yt_offset = yb_offset + yt_row_offset * params.k + yt_col_index;
        var y_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (yt_row_offset + i < params.n && yt_col_index < params.k) {
                let y_index = yt_offset + i * params.k;
                y_tmp[i] = y[y_index];
            }
        }
        tile_y[tx * 16u + ty] = y_tmp;
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            let x_val = tile_x[ty * 16u + t];
            let y_val = tile_y[t * 16u + tx];
            acc0 = fma(vec4<f32>(x_val[0u]), y_val, acc0);
            acc1 = fma(vec4<f32>(x_val[1u]), y_val, acc1);
            acc2 = fma(vec4<f32>(x_val[2u]), y_val, acc2);
            acc3 = fma(vec4<f32>(x_val[3u]), y_val, acc3);
        }
        workgroupBarrier();
    }

    let rb_offset = batch * params.m * params.n;
    let acc = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);

    for (var ri = 0u; ri < 4u; ri++) {
        let row_index = row_vec_index + ri;
        if (batch < params.b && row_index < params.m) {
            let offset = rb_offset + row_index * params.n;
            for (var ci = 0u; ci < 4u; ci++) {
                let col_index = col_vec_index + ci;
                if (col_index < params.n) {
                    result[offset + col_index] = acc[ri][ci];
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn mat_mul_tn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row_vec_index = global_id.y * 4u;
    let col_vec_index = global_id.x * 4u;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = ty * 16u + tx;

    let xb_offset = batch * params.k * params.m;
    let yb_offset = batch * params.k * params.n;

    var acc0 = vec4<f32>(0f);
    var acc1 = vec4<f32>(0f);
    var acc2 = vec4<f32>(0f);
    var acc3 = vec4<f32>(0f);
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_offset = xb_offset + xi * params.m + row_vec_index;
        var x_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (row_vec_index + i < params.m && xi < params.k) {
                let x_index = x_offset + i;
                x_tmp[i] = x[x_index];
            }
        }
        tile_x[t_index] = x_tmp;

        let yi = step + ty;
        let y_offset = yb_offset + yi * params.n + col_vec_index;
        var y_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (yi < params.k && col_vec_index + i < params.n) {
                let y_index = y_offset + i;
                y_tmp[i] = y[y_index];
            }
        }
        tile_y[t_index] = y_tmp;
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            let x_val = tile_x[ty * 16u + t];
            let y_val = tile_y[t * 16u + tx];
            acc0 = fma(vec4<f32>(x_val[0u]), y_val, acc0);
            acc1 = fma(vec4<f32>(x_val[1u]), y_val, acc1);
            acc2 = fma(vec4<f32>(x_val[2u]), y_val, acc2);
            acc3 = fma(vec4<f32>(x_val[3u]), y_val, acc3);
        }
        workgroupBarrier();
    }

    let rb_offset = batch * params.m * params.n;
    let acc = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);

    for (var ri = 0u; ri < 4u; ri++) {
        let row_index = row_vec_index + ri;
        if (batch < params.b && row_index < params.m) {
            let offset = rb_offset + row_index * params.n;
            for (var ci = 0u; ci < 4u; ci++) {
                let col_index = col_vec_index + ci;
                if (col_index < params.n) {
                    result[offset + col_index] = acc[ri][ci];
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn mat_mul_tt(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let batch = global_id.z;
    let row_vec_index = global_id.y * 4u;
    let col_vec_index = global_id.x * 4u;

    let tx = local_id.x;
    let ty = local_id.y;
    let t_index = ty * 16u + tx;

    let xb_offset = batch * params.k * params.m;
    let yb_offset = batch * params.n * params.k;

    // (64, 16)の領域をコピー
    // 縦方向に4つずつ取っていくため、Y軸方向の割当は64/4=16
    let yt_x = tx;
    let yt_y_vec = ty * 4u;
    let yt_row_offset = workgroup_id.x * 64u + yt_y_vec;

    var acc0 = vec4<f32>(0f);
    var acc1 = vec4<f32>(0f);
    var acc2 = vec4<f32>(0f);
    var acc3 = vec4<f32>(0f);
    for (var step = 0u; step < params.k; step += 16u) {
        let xi = step + tx;
        let x_offset = xb_offset + xi * params.m + row_vec_index;
        var x_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (row_vec_index + i < params.m && xi < params.k) {
                let x_index = x_offset + i;
                x_tmp[i] = x[x_index];
            }
        }
        tile_x[t_index] = x_tmp;

        let yt_col_index = step + yt_x;
        let yt_offset = yb_offset + yt_row_offset * params.k + yt_col_index;
        var y_tmp = vec4<f32>(0f);
        for (var i = 0u; i < 4u; i++) {
            if (yt_row_offset + i < params.n && yt_col_index < params.k) {
                let y_index = yt_offset + i * params.k;
                y_tmp[i] = y[y_index];
            }
        }
        tile_y[tx * 16u + ty] = y_tmp;
        workgroupBarrier();

        for (var t = 0u; t < 16u; t++) {
            let x_val = tile_x[ty * 16u + t];
            let y_val = tile_y[t * 16u + tx];
            acc0 = fma(vec4<f32>(x_val[0u]), y_val, acc0);
            acc1 = fma(vec4<f32>(x_val[1u]), y_val, acc1);
            acc2 = fma(vec4<f32>(x_val[2u]), y_val, acc2);
            acc3 = fma(vec4<f32>(x_val[3u]), y_val, acc3);
        }
        workgroupBarrier();
    }

    let rb_offset = batch * params.m * params.n;
    let acc = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);

    for (var ri = 0u; ri < 4u; ri++) {
        let row_index = row_vec_index + ri;
        if (batch < params.b && row_index < params.m) {
            let offset = rb_offset + row_index * params.n;
            for (var ci = 0u; ci < 4u; ci++) {
                let col_index = col_vec_index + ci;
                if (col_index < params.n) {
                    result[offset + col_index] = acc[ri][ci];
                }
            }
        }
    }
}
