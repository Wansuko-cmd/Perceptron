use wgpu::{Device, util::DeviceExt};

use crate::{kernels::task::ComputeTask, resource::buffer::GPUBuffer};

pub struct Shape {
    device: Device,

    slice: wgpu::ComputePipeline,
    slice_bgl: wgpu::BindGroupLayout,

    transpose_d2: wgpu::ComputePipeline,
    transpose_d4: wgpu::ComputePipeline,
    transpose_bgl: wgpu::BindGroupLayout,
}

impl Shape {
    const WORKGROUP_SIZE: u32 = 256;

    pub fn new(device: &Device) -> Self {
        let slice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shape::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("slice.wgsl").into()),
        });

        let slice_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shape::new"),
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

        let transpose_d2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shape::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transpose_d2.wgsl").into()),
        });
        let transpose_d4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shape::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transpose_d4.wgsl").into()),
        });

        let transpose_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shape::new"),
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
            label: Some("Shape::new"),
            bind_group_layouts: &[&transpose_bgl],
            immediate_size: 0,
        });

        Shape {
            device: device.clone(),
            slice: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slice"),
                layout: Some(&pipeline_layout),
                module: &slice_shader,
                entry_point: Some("slice"),
                compilation_options: Default::default(),
                cache: None,
            }),
            slice_bgl: slice_bgl,
            transpose_d2: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transpose_d2"),
                layout: Some(&pipeline_layout),
                module: &transpose_d2_shader,
                entry_point: Some("transpose_d2"),
                compilation_options: Default::default(),
                cache: None,
            }),
            transpose_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transpose_d4"),
                layout: Some(&pipeline_layout),
                module: &transpose_d4_shader,
                entry_point: Some("transpose_d4"),
                compilation_options: Default::default(),
                cache: None,
            }),
            transpose_bgl: transpose_bgl,
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

impl Shape {
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
            layout: &self.slice_bgl,
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

        let workgroups = result.workgroup_count(Shape::WORKGROUP_SIZE);

        ComputeTask {
            label: label,
            pipeline: &self.slice,
            bind_group: bind_group,
            workgroups: workgroups,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeD2Params {
    oi: u32,
    oj: u32,
    _pad1: u32,
    _pad2: u32,
}

impl Shape {
    pub fn transpose_d2<'a>(
        &'a self,
        x: &GPUBuffer,
        xi: usize, xj: usize,
        result: &GPUBuffer,
    ) -> ComputeTask<'a> {
        let label = "transpose_d2";
        let device = &self.device;
        let params = TransposeD2Params {
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
            layout: &self.transpose_bgl,
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

        let workgroups = result.workgroup_count(Shape::WORKGROUP_SIZE);

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
struct TransposeD4Params {
    new_stride: [u32; 4],
    permuted_stride: [u32; 4],
}

impl Shape {
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
        let params = TransposeD4Params {
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
            layout: &self.transpose_bgl,
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

        let workgroups = result.workgroup_count(Shape::WORKGROUP_SIZE);

        ComputeTask {
            label: label,
            pipeline: &self.transpose_d4,
            bind_group: bind_group,
            workgroups: workgroups,
        }
    }
}
