use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

pub fn transpose_d2(x: &GPUBuffer, xi: usize, xj: usize, result: &GPUBuffer, context: &Context) {
    transpose(
        "transpose_d2",
        x, 1, 1,  xi, xj,
        0, 1,  3, 2,
        result,
        context,
    );
}

pub fn transpose_d3(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize,
    axis_i: usize, axis_j: usize, axis_k: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    transpose(
        "transpose_d3",
        x, 1,  xi, xj, xk,
        0, axis_i + 1,  axis_j + 1, axis_k + 1,
        result,
        context,
    );
}

pub fn transpose_d4(
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    transpose(
        "transpose_d4",
        x,  xi, xj, xk, xl,
        axis_i,  axis_j, axis_k, axis_l,
        result,
        context,
    );
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeParams {
    new_stride: [u32; 4],
    permuted_stride: [u32; 4],
    length: u32,
    _pad: [u32; 3],
}

fn transpose(
    label: &str,
    x: &GPUBuffer,
    xi: usize, xj: usize, xk: usize, xl: usize,
    axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let device = &context.device;
    let queue = &context.queue;
    let pipeline = &context.pipeline;

    let old_shape = [xi, xj, xk, xl];
    let new_shape = [old_shape[axis_i], old_shape[axis_j], old_shape[axis_k], old_shape[axis_l]];

    let old_stride = [old_shape[1] * old_shape[2] * old_shape[3], old_shape[2] * old_shape[3], old_shape[3], 1];
    let new_stride = [new_shape[1] * new_shape[2] * new_shape[3], new_shape[2] * new_shape[3], new_shape[3], 1];
    let permuted_stride = [old_stride[axis_i], old_stride[axis_j], old_stride[axis_k], old_stride[axis_l]];
    let param = TransposeParams {
        new_stride: new_stride.map(|v| v as u32),
        permuted_stride: permuted_stride.map(|v| v as u32),
        length: result.count() as u32,
        _pad: [0; 3],
    };

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&param),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.transpose.bind_group_layout,
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

        compute_pass.set_pipeline(&pipeline.transpose.transpose_d4);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(x.workgroup_count(WORKGROUP_SIZE), 1, 1);
    }

    queue.submit(Some(encoder.finish()));
}
