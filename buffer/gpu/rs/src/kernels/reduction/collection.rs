use wgpu::{Device, util::DeviceExt};

use crate::{kernels::task::ComputeTask, resource::buffer::GPUBuffer};

pub struct Collection {
    device: Device,

    average_d3: wgpu::ComputePipeline,
    max_d3: wgpu::ComputePipeline,
    min_d3: wgpu::ComputePipeline,
    sum_d3: wgpu::ComputePipeline,

    bind_group_layout: wgpu::BindGroupLayout,
}

impl Collection {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Collection::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("collection_d3.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collection::new"),
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
            label: Some("Collection::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Collection {
            device: device.clone(),
            average_d3: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("average_d3"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("average_d3"),
                compilation_options: Default::default(),
                cache: None,
            }),
            max_d3: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("max_d3"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("max_d3"),
                compilation_options: Default::default(),
                cache: None,
            }),
            min_d3: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("min_d3"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("min_d3"),
                compilation_options: Default::default(),
                cache: None,
            }),
            sum_d3: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sum_d3"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("sum_d3"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    xi: u32,
    xj: u32,
    xk: u32,
    _pad: u32,
}

impl Collection {
    const WORKGROUP_SIZE: u32 = 256;

    pub fn average_d3<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "average_d3",
            &self.average_d3,
            x,
            xi, xj, xk,
            result,
        )
    }

    pub fn max_d3<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "max_d3",
            &self.max_d3,
            x,
            xi, xj, xk,
            result,
        )
    }

    pub fn min_d3<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "min_d3",
            &self.min_d3,
            x,
            xi, xj, xk,
            result,
        )
    }

    pub fn sum_d3<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "sum_d3",
            &self.sum_d3,
            x,
            xi, xj, xk,
            result,
        )
    }

    fn create_task<'a>(
        &'a self,
        label: &'static str,
        pipeline: &'a wgpu::ComputePipeline,
        x: &GPUBuffer,
        xi: usize, xj: usize, xk: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        let device = &self.device;
        let params = Params { xi: xi as u32, xj: xj as u32, xk: xk as u32, _pad: 0 };
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

        let workgroups = result.workgroup_count(Collection::WORKGROUP_SIZE);

        ComputeTask {
            label: label,
            pipeline: pipeline,
            bind_group: bind_group,
            workgroups: workgroups,
        }
    }
}
