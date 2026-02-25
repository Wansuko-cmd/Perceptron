use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

pub fn mat_mul(
    x: &GPUBuffer, trans_x: bool,
    y: &GPUBuffer, trans_y: bool,
    m: usize, n: usize, k: usize, b: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let label = "mat_mul";
    let device = &context.device;
    let queue = &context.queue;
    let pipeline = match (trans_x, trans_y) {
        (false, false) => &context.pipeline.mat_mul.mat_mul_nn,
        (false, true) => &context.pipeline.mat_mul.mat_mul_nt,
        (true, false) => &context.pipeline.mat_mul.mat_mul_tn,
        (true, true) => &context.pipeline.mat_mul.mat_mul_tt,
    };

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&[m, n, k, b].map(|v| v as u32)),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &context.pipeline.mat_mul.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
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
