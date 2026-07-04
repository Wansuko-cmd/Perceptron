struct Params {
    xi: u32,
    xj: u32,
    b: u32,
    window: u32,
    stride: u32,
    padding: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn unfold_d1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let rb = params.b;
    let ri = params.xi;
    let rj = (params.xj - params.window + params.padding * 2) / params.stride + 1;
    let rk = params.window;

    let result_index = id.y * num_groups.x * 256u + id.x;
    if (result_index >= rb * ri * rj * rk) {
        return;
    }

    var tmp = result_index;
    let nk = tmp % rk; tmp /= rk;
    let nj = tmp % rj; tmp /= rj;
    let ni = tmp % ri; tmp /= ri;
    let nb = tmp;

    let ob = nb;
    let oi = ni;
    let oj = nj * params.stride + nk - params.padding;
    if (oj < params.xj) {
        let x_index = (ob * params.xi + oi) * params.xj + oj;
        result[result_index] = x[x_index];
    }
}
