use wgpu::{BindGroupLayout, Device, wgc::device};

pub struct Pipeline {
    pub math: Math,
    pub transpose: Transpose,
}

impl Pipeline {
    pub fn new(device: &Device) -> Self {
        Pipeline {
            math: Math::new(device),
            transpose: Transpose::new(device),
        }
    }
}

pub struct Math {
    pub exp_d1: wgpu::ComputePipeline,
    pub ln_d1: wgpu::ComputePipeline,
    pub pow_d1: wgpu::ComputePipeline,
    pub sqrt_d1: wgpu::ComputePipeline,

    pub bind_group_layout: BindGroupLayout,
    pub bind_group_layout_with_param: BindGroupLayout,
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

        let bind_group_layout_with_param = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("Math::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline_layout_with_param = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Math::new"),
            bind_group_layouts: &[&bind_group_layout_with_param],
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
            ln_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ln_d1"),
                layout: Some(&pipeline_layout_with_param),
                module: &shader,
                entry_point: Some("ln_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            pow_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pow_d1"),
                layout: Some(&pipeline_layout_with_param),
                module: &shader,
                entry_point: Some("pow_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            sqrt_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sqrt_d1"),
                layout: Some(&pipeline_layout_with_param),
                module: &shader,
                entry_point: Some("sqrt_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
            bind_group_layout_with_param: bind_group_layout_with_param,
        }
    }
}

pub struct Transpose {
    pub transpose_d4: wgpu::ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
}

impl Transpose {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Transpose::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("transpose.wgsl").into()),
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
            transpose_d4: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transpose_d4"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("transpose_d4"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
        }
    }
}
