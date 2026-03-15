use wgpu::{Device, util::DeviceExt};

use crate::{kernels::task::ComputeTask, resource::buffer::GPUBuffer};

pub struct Buffer {
    device: Device,
    slice: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Buffer {
    const WORKGROUP_SIZE: u32 = 256;

    pub fn new(device: &Device) -> Self {
        let shader_slice = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Buffer::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("slice.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Buffer::new"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Buffer::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Buffer {
            device: device.clone(),
            slice: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slice"),
                layout: Some(&pipeline_layout),
                module: &shader_slice,
                entry_point: Some("slice"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceParams {
    start: u32,
    step: i32,
    _pad1: u32,
    _pad2: u32,
}

impl Buffer {
    pub fn slice<'a>(
        &'a self,
        x: &GPUBuffer,
        start: usize, step: isize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        let label = "slice";
        let device = &self.device;
        let params = SliceParams {
            start: start as u32,
            step: step as i32,
            _pad1: 0u32,
            _pad2: 0u32,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ]
        });

        let workgroups = result.workgroup_count(Buffer::WORKGROUP_SIZE);

        ComputeTask {
            label: label,
            pipeline: &self.slice,
            bind_group: bind_group,
            workgroups: workgroups,
        }
    }
}
