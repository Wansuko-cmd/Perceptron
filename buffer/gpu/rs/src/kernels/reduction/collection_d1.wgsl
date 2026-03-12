@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> cache: array<f32, 256>;

const F32_MAX: f32 = 3.402823466e+38;
const F32_MIN: f32 = -3.402823466e+38;

@compute @workgroup_size(256)
fn min_d1(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let global_index = global_id.x;
    let local_index = local_id.x;

    var acc = F32_MAX;
    var i = global_index;
    while (i < arrayLength(&x)) {
        acc = min(acc, x[i]);
        i += num_groups.x * 256u;
    }
    cache[local_index] = acc;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_index < s) {
            cache[local_index] = min(cache[local_index], cache[local_index + s]);
        }
        workgroupBarrier();
    }

    if (local_index == 0u && group_id.x < arrayLength(&result)) {
        result[group_id.x] = cache[0];
    }
}

@compute @workgroup_size(256)
fn max_d1(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let global_index = global_id.x;
    let local_index = local_id.x;

    var acc = F32_MIN;
    var i = global_index;
    while (i < arrayLength(&x)) {
        acc = max(acc, x[i]);
        i += num_groups.x * 256u;
    }
    cache[local_index] = acc;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_index < s) {
            cache[local_index] = max(cache[local_index], cache[local_index + s]);
        }
        workgroupBarrier();
    }

    if (local_index == 0u && group_id.x < arrayLength(&result)) {
        result[group_id.x] = cache[0];
    }
}

@compute @workgroup_size(256)
fn sum_d1(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let global_index = global_id.x;
    let local_index = local_id.x;

    var acc = 0f;
    var i = global_index;
    while (i < arrayLength(&x)) {
        acc += x[i];
        i += num_groups.x * 256u;
    }
    cache[local_index] = acc;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_index < s) {
            cache[local_index] += cache[local_index + s];
        }
        workgroupBarrier();
    }

    if (local_index == 0u && group_id.x < arrayLength(&result)) {
        result[group_id.x] = cache[0];
    }
}

@compute @workgroup_size(256)
fn average_d1(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let global_index = global_id.x;
    let local_index = local_id.x;

    var acc = 0f;
    var i = global_index;
    while (i < arrayLength(&x)) {
        acc += x[i];
        i += num_groups.x * 256u;
    }
    cache[local_index] = acc;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_index < s) {
            cache[local_index] += cache[local_index + s];
        }
        workgroupBarrier();
    }

    if (local_index == 0u && group_id.x < arrayLength(&result)) {
        if (num_groups.x == 1u) {
            result[0] = cache[0] / f32(arrayLength(&x));
        } else {
            result[group_id.x] = cache[0];
        }
    }
}
