use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

pub fn exp_d1(x: &GPUBuffer, result: &GPUBuffer, context: &Context) {
    let device = &context.device;
    let queue = &context.queue;
    let pipeline = &context.pipeline;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exp_d1"),
        layout: &pipeline.math.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result.buffer.as_entire_binding(),
            },
        ]
    });

    let mut encoder = device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
        label: Some("exp_d1")
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exp_d1"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline.math.exp_d1);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(x.workgroup_count(WORKGROUP_SIZE), 1, 1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn ln_d1(x: &GPUBuffer, e: f32, result: &GPUBuffer, context: &Context) {
    dispatch_with_param(&context.pipeline.math.ln_d1, "ln_d1", x,e, result, context);
}

pub fn pow_d1(x: &GPUBuffer, n: i32, result: &GPUBuffer, context: &Context) {
    dispatch_with_param(&context.pipeline.math.pow_d1, "pow_d1", x,n as f32, result, context);
}

pub fn sqrt_d1(x: &GPUBuffer, e: f32, result: &GPUBuffer, context: &Context) {
    dispatch_with_param(&context.pipeline.math.sqrt_d1, "sqrt_d1", x,e, result, context);
}

fn dispatch_with_param(
    pipeline: &wgpu::ComputePipeline, label: &str,
    x: &GPUBuffer, param: f32, result: &GPUBuffer,
    context: &Context,
) {
    let device = &context.device;
    let queue = &context.queue;

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&[param, 0.0, 0.0, 0.0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &context.pipeline.math.bind_group_layout_with_param,
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

        compute_pass.dispatch_workgroups(x.workgroup_count(WORKGROUP_SIZE), 1, 1);
    }

    queue.submit(Some(encoder.finish()));
}
