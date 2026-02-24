use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

pub fn average_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d1(
        &context.pipeline.collection.average_d3,
        "average_d1",
        x,
        result,
        context,
    );
}

pub fn average_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d2(
        &context.pipeline.collection.average_d3,
        "average_d2",
        x, xi, xj, 
        axis,
        result,
        context,
    );
}

pub fn average_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d3(
        &context.pipeline.collection.average_d3,
        "average_d3",
        x, xi, xj, xk,
        axis,
        result,
        context,
    );
}

pub fn max_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d1(
        &context.pipeline.collection.max_d3,
        "max_d1",
        x,
        result,
        context,
    );
}

pub fn max_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d2(
        &context.pipeline.collection.max_d3,
        "max_d2",
        x, xi, xj, 
        axis,
        result,
        context,
    );
}

pub fn max_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d3(
        &context.pipeline.collection.max_d3,
        "max_d3",
        x, xi, xj, xk,
        axis,
        result,
        context,
    );
}

pub fn min_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d1(
        &context.pipeline.collection.min_d3,
        "min_d1",
        x,
        result,
        context,
    );
}

pub fn min_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d2(
        &context.pipeline.collection.min_d3,
        "min_d2",
        x, xi, xj, 
        axis,
        result,
        context,
    );
}

pub fn min_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d3(
        &context.pipeline.collection.min_d3,
        "min_d3",
        x, xi, xj, xk,
        axis,
        result,
        context,
    );
}

pub fn sum_d1(
    x: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d1(
        &context.pipeline.collection.sum_d3,
        "sum_d1",
        x,
        result,
        context,
    );
}

pub fn sum_d2(
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d2(
        &context.pipeline.collection.sum_d3,
        "sum_d2",
        x, xi, xj, 
        axis,
        result,
        context,
    );
}

pub fn sum_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    reduce_d3(
        &context.pipeline.collection.sum_d3,
        "sum_d3",
        x, xi, xj, xk,
        axis,
        result,
        context,
    );
}

fn reduce_d1(
    pipeline: &wgpu::ComputePipeline, label: &str,
    x: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    _reduce_d3(
        pipeline,
        label,
        x, 1, x.count(), 1,
        result,
        context,
    );
}

fn reduce_d2(
    pipeline: &wgpu::ComputePipeline, label: &str,
    x: &GPUBuffer,
    xi: usize, xj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    match axis {
        0 => {
            _reduce_d3(
                pipeline,
                label,
                x, 1, xi, xj,
                result,
                context,
            );
        },
        _ => {
            _reduce_d3(
                pipeline,
                label,
                x, xi, xj, 1,
                result,
                context,
            );
        }
    }
}

fn reduce_d3(
    pipeline: &wgpu::ComputePipeline, label: &str,
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    match axis {
        0 => {
            _reduce_d3(
                pipeline,
                label,
                x, 1, xi, xj * xk,
                result,
                context,
            );
        },
        1 => {
            _reduce_d3(
                pipeline,
                label,
                x, xi, xj, xk,
                result,
                context,
            );
        }
        _ => {
            _reduce_d3(
                pipeline,
                label,
                x, xi * xj, xk, 1,
                result,
                context,
            );
        }
    }
}

fn _reduce_d3(
    pipeline: &wgpu::ComputePipeline, label: &str,
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let device = &context.device;
    let queue = &context.queue;

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&[xi, xj, xk, result.count()].map(|v| v as u32)),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &context.pipeline.collection.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
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
