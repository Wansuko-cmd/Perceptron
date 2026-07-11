use wgpu::Device;

use crate::{kernels::task::ComputeTask, resource::buffer::GPUBuffer};

pub struct Operation {
    device: Device,

    plus_d4: wgpu::ComputePipeline,
    minus_d4: wgpu::ComputePipeline,
    times_d4: wgpu::ComputePipeline,
    div_d4: wgpu::ComputePipeline,

    bind_group_layout: wgpu::BindGroupLayout,
}

impl Operation {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Operation::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("operation_d4.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Operation::new"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Operation::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: std::mem::size_of::<Params>() as u32,
        });

        Operation {
            device: device.clone(),
            plus_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("plus_d4"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("plus_d4"),
                compilation_options: Default::default(),
                cache: None,
            }),
            minus_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("minus_d4"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("minus_d4"),
                compilation_options: Default::default(),
                cache: None,
            }),
            times_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("times_d4"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("times_d4"),
                compilation_options: Default::default(),
                cache: None,
            }),
            div_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("div_d4"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("div_d4"),
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
    result_shape: [u32; 4],
    x_stride: [u32; 4],
    y_stride: [u32; 4],
}

impl Operation {
    const WORKGROUP_SIZE: u32 = 256;

    pub fn plus_d4<'a>(
        &'a self,
        x: &GPUBuffer,
        y: &GPUBuffer,
        result_shape: [u32; 4],
        x_stride: [u32; 4],
        y_stride: [u32; 4],
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "plus_d4",
            &self.plus_d4,
            x,
            y,
            &Params { result_shape: result_shape, x_stride: x_stride, y_stride: y_stride },
            result,
        )
    }

    pub fn minus_d4<'a>(
        &'a self,
        x: &GPUBuffer,
        y: &GPUBuffer,
        result_shape: [u32; 4],
        x_stride: [u32; 4],
        y_stride: [u32; 4],
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "minus_d4",
            &self.minus_d4,
            x,
            y,
            &Params { result_shape: result_shape, x_stride: x_stride, y_stride: y_stride },
            result,
        )
    }

    pub fn times_d4<'a>(
        &'a self,
        x: &GPUBuffer,
        y: &GPUBuffer,
        result_shape: [u32; 4],
        x_stride: [u32; 4],
        y_stride: [u32; 4],
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "times_d4",
            &self.times_d4,
            x,
            y,
            &Params { result_shape: result_shape, x_stride: x_stride, y_stride: y_stride },
            result,
        )
    }

    pub fn div_d4<'a>(
        &'a self,
        x: &GPUBuffer,
        y: &GPUBuffer,
        result_shape: [u32; 4],
        x_stride: [u32; 4],
        y_stride: [u32; 4],
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        self.create_task(
            "div_d4",
            &self.div_d4,
            x,
            y,
            &Params { result_shape: result_shape, x_stride: x_stride, y_stride: y_stride },
            result,
        )
    }

    fn create_task<'a>(
        &'a self,
        label: &'static str,
        pipeline: &'a wgpu::ComputePipeline,
        x: &GPUBuffer,
        y: &GPUBuffer,
        params: &Params,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        let device = &self.device;

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
                    resource: y.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.buffer.as_entire_binding(),
                },
            ]
        });

        let workgroups = result.workgroup_count(Operation::WORKGROUP_SIZE);

        ComputeTask {
            label: label,
            pipeline: pipeline,
            bind_group: bind_group,
            workgroups: workgroups,
            params: Some(bytemuck::bytes_of(params).to_vec()),
        }
    }
}

