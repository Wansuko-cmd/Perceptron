use wgpu::{BindGroupLayout, Device};

pub struct Pipeline {
    pub math: Math,
}

impl Pipeline {
    pub fn new(device: &Device) -> Self {
        Pipeline { math: Math::new(device) }
    }
}

pub struct Math {
    pub exp_d1: wgpu::ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
}

impl Math {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Math::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("math.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Math::new"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Math::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Math {
            exp_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("exp_d1"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("exp_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
        }
    }
}
