struct Params {
    // fromが予約語
    lower: f32,
    upper: f32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64, 1)
fn random_d1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {
    let stride = num_groups.x * 64;
    let index = id.y * stride + id.x;
    if (index >= arrayLength(&result)) {
        return;
    }

    let seed = index + params.seed;
    var state = seed * 747796405u + 2891336453u;
    var tmp = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    tmp = (tmp >> 22u) ^ tmp;

    let unit = f32(tmp) / 4294967295.0;
    result[index] = params.lower + unit * (params.upper - params.lower);
}
