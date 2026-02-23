use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

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
