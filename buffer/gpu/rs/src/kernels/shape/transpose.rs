use wgpu::{Device, util::DeviceExt};

use crate::{kernels::task::ComputeTask, resource::buffer::GPUBuffer};

pub struct Transpose {
    device: Device,
    transpose_d2: wgpu::ComputePipeline,
    transpose_d4: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Transpose {
    pub fn new(device: &Device) -> Self {
        let shader_d2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Transpose::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transpose_d2.wgsl").into()),
        });
        let shader_d4 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Transpose::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transpose_d4.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Transpose::new"),
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
            label: Some("Transpose::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Transpose {
            device: device.clone(),
            transpose_d2: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transpose_d2"),
                layout: Some(&pipeline_layout),
                module: &shader_d2,
                entry_point: Some("transpose_d2"),
                compilation_options: Default::default(),
                cache: None,
            }),
            transpose_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transpose_d4"),
                layout: Some(&pipeline_layout),
                module: &shader_d4,
                entry_point: Some("transpose_d4"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct D2Params {
    oi: u32,
    oj: u32,
    _pad1: u32,
    _pad2: u32,
}

impl Transpose {
    pub fn transpose_d2<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        let label = "transpose_d2";
        let device = &self.device;
        let params = D2Params {
            oi: xi as u32,
            oj: xj as u32,
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

        let workgroups = [((xj + 15) / 16) as u32, ((xi + 15) / 16) as u32, 1];

        ComputeTask {
            label: label,
            pipeline: &self.transpose_d2,
            bind_group: bind_group,
            workgroups: workgroups,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct D4Params {
    new_stride: [u32; 4],
    permuted_stride: [u32; 4],
}

impl Transpose {
    const WORKGROUP_SIZE: u32 = 256;

    pub fn transpose_d3<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize,
        axis_i: usize, axis_j: usize, axis_k: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "transpose_d3",
            x, 1,  xi, xj, xk,
            0, axis_i + 1,  axis_j + 1, axis_k + 1,
            result,
        )
    }

    pub fn transpose_d4<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize, xl: usize,
        axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "transpose_d4",
            x,  xi, xj, xk, xl,
            axis_i,  axis_j, axis_k, axis_l,
            result,
        )
    }

    fn create_task<'a>(
        &'a self,
        label: &'static str,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize, xl: usize,
        axis_i: usize, axis_j: usize, axis_k: usize, axis_l: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        let device = &self.device;
        let old_shape = [xi, xj, xk, xl];
        let new_shape = [old_shape[axis_i], old_shape[axis_j], old_shape[axis_k], old_shape[axis_l]];

        let old_stride = [old_shape[1] * old_shape[2] * old_shape[3], old_shape[2] * old_shape[3], old_shape[3], 1];
        let new_stride = [new_shape[1] * new_shape[2] * new_shape[3], new_shape[2] * new_shape[3], new_shape[3], 1];
        let permuted_stride = [old_stride[axis_i], old_stride[axis_j], old_stride[axis_k], old_stride[axis_l]];
        let params = D4Params {
            new_stride: new_stride.map(|v| v as u32),
            permuted_stride: permuted_stride.map(|v| v as u32),
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

        let workgroups = result.workgroup_count(Transpose::WORKGROUP_SIZE);

        ComputeTask {
            label: label,
            pipeline: &self.transpose_d4,
            bind_group: bind_group,
            workgroups: workgroups,
        }
    }
}
