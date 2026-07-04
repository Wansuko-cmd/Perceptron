struct Params {
    i: u32,
    j: u32,
    k: u32,
    b: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<atomic<u32>>;
var<immediate> params: Params;

@compute @workgroup_size(256, 1)
fn scatter_add(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 256;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&x)) {
        return;
    }

    let n = arrayLength(&y) / params.b;
    var tmp = index;
    let ok = tmp % params.k; tmp = tmp / params.k;
    let oj = tmp % n; tmp = tmp / n;
    let oi = tmp % params.i; tmp = tmp / params.i;
    let ob = tmp;

    let y_val = u32(y[ob * n + oj]);
    let result_index = (oi * params.j + y_val) * params.k + ok;
    atomicAdd(result_index, x[index]);
}

fn atomicAdd(index: u32, value: f32) {
    var old_bits = atomicLoad(&result[index]);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_value = old_value + value;
        let new_bits = bitcast<u32>(new_value);
        
        let res = atomicCompareExchangeWeak(&result[index], old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
        old_bits = res.old_value;
    }
}
