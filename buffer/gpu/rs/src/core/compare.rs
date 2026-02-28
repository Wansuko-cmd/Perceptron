use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompareParams {
    val1: f32,
    val2: f32,
    _pad: [u32; 2],
}

pub fn gt_d1_to_d0(
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.gt_d1_to_d0,
        "gt_d1_to_d0",
        // condition, yは使わない
        x, x, x,
        CompareParams { val1: y, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn gt_d1_to_d1(
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.gt_d1_to_d1,
        "gt_d1_to_d1",
        // conditionは使わない
        x, x, y,
        CompareParams { val1: 0f32, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn lt_d1_to_d0(
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.lt_d1_to_d0,
        "lt_d1_to_d0",
       // condition, yは使わない
        x, x, x,
        CompareParams { val1: y, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn lt_d1_to_d1(
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.lt_d1_to_d1,
        "lt_d1_to_d1",
        // conditionは使わない
        x, x, y,
        CompareParams { val1: 0f32, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn where_d0_to_d0(
    condition: &GPUBuffer,
    x: f32,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.where_d0_to_d0,
        "condition_d0_to_d0",
        // x, yは使わない
        condition, condition, condition,
        CompareParams { val1: x, val2: y, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn where_d0_to_d1(
    condition: &GPUBuffer,
    x: f32,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.where_d0_to_d1,
        "condition_d0_to_d1",
        // xは使わない
        condition, y, y,
        CompareParams { val1: x, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn where_d1_to_d0(
    condition: &GPUBuffer,
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.where_d1_to_d0,
        "condition_d1_to_d1",
       // yは使わない
        condition, x, x,
        CompareParams { val1: y, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

pub fn where_d1_to_d1(
    condition: &GPUBuffer,
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    compare_execute(
        &context.pipeline.compare.where_d1_to_d1,
        "condition_d1_to_d1",
        condition, x, y,
        CompareParams { val1: 0f32, val2: 0f32, _pad: [0; 2] },
        result,
        context,
    );
}

fn compare_execute(
    pipeline: &wgpu::ComputePipeline, label: &str,
    condition: &GPUBuffer,
    x: &GPUBuffer,
    y: &GPUBuffer,
    params: CompareParams,
    result: &GPUBuffer,
    context: &Context,
) {
    let device = &context.device;
    let queue = &context.queue;

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &context.pipeline.compare.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: condition.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: param_buffer.as_entire_binding(),
            },
        ]
    });

    let mut encoder = device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
        label: Some(label)
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(result.workgroup_count(WORKGROUP_SIZE), 1, 1);
    }

    queue.submit(Some(encoder.finish()));
}
