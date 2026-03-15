struct Params {
    oi: u32,
    oj: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn transpose_d2(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let old_i = group_id.y * 16u + local_id.y;
    let old_j = group_id.x * 16u + local_id.x;
    if (old_i < params.oi && old_j < params.oj) {
        let old_index = old_i * params.oj + old_j;
        tile[local_id.y][local_id.x] = x[old_index];
    }

    workgroupBarrier();
    
    let new_i = group_id.x * 16u + local_id.y;
    let new_j = group_id.y * 16u + local_id.x;
    if (new_i < params.oj && new_j < params.oi) {
        let new_index = new_i * params.oi + new_j;
        result[new_index] = tile[local_id.x][local_id.y];
    }
}
