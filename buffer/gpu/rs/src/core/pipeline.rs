use wgpu::{BindGroupLayout, Device};

pub struct Pipeline {
    pub collection: Collection,
    pub compare: Compare,
    pub math: Math,
    pub mat_mul: MatMul,
    pub operation: Operation,
    pub transpose: Transpose,
}

impl Pipeline {
    pub fn new(device: &Device) -> Self {
        Pipeline {
            collection: Collection::new(device),
            compare: Compare::new(device),
            math: Math::new(device),
            mat_mul: MatMul::new(device),
            operation: Operation::new(device),
            transpose: Transpose::new(device),
        }
    }
}

pub struct Collection {
    pub average_d3: wgpu::ComputePipeline,
    pub max_d3: wgpu::ComputePipeline,
    pub min_d3: wgpu::ComputePipeline,
    pub sum_d3: wgpu::ComputePipeline,

    pub bind_group_layout: BindGroupLayout,
}

impl Collection {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Collection::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("collection.wgsl").into()),
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

pub struct Compare {
    pub gt_d1_to_d0: wgpu::ComputePipeline,
    pub gt_d1_to_d1: wgpu::ComputePipeline,
    pub lt_d1_to_d0: wgpu::ComputePipeline,
    pub lt_d1_to_d1: wgpu::ComputePipeline,
    pub where_d0_to_d0: wgpu::ComputePipeline,
    pub where_d0_to_d1: wgpu::ComputePipeline,
    pub where_d1_to_d0: wgpu::ComputePipeline,
    pub where_d1_to_d1: wgpu::ComputePipeline,

    pub bind_group_layout: BindGroupLayout,
}

impl Compare {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compare::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compare.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compare::new"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Compare::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Compare {
            gt_d1_to_d0: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gt_d1_to_d0"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("gt_d1_to_d0"),
                compilation_options: Default::default(),
                cache: None,
            }),
            gt_d1_to_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gt_d1_to_d1"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("gt_d1_to_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            lt_d1_to_d0: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lt_d1_to_d0"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("lt_d1_to_d0"),
                compilation_options: Default::default(),
                cache: None,
            }),
            lt_d1_to_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lt_d1_to_d1"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("lt_d1_to_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            where_d0_to_d0: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("where_d0_to_d0"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("where_d0_to_d0"),
                compilation_options: Default::default(),
                cache: None,
            }),
            where_d0_to_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("where_d0_to_d1"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("where_d0_to_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            where_d1_to_d0: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("where_d1_to_d0"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("where_d1_to_d0"),
                compilation_options: Default::default(),
                cache: None,
            }),
            where_d1_to_d1: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("where_d1_to_d1"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("where_d1_to_d1"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
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

pub struct MatMul {
    pub mat_mul_nn: wgpu::ComputePipeline,
    pub mat_mul_nt: wgpu::ComputePipeline,
    pub mat_mul_tn: wgpu::ComputePipeline,
    pub mat_mul_tt: wgpu::ComputePipeline,

    pub bind_group_layout: BindGroupLayout,
}

impl MatMul {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mat_mul.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul::new"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("MatMul::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        MatMul {
            mat_mul_nn: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mat_mul_nn"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("mat_mul_nn"),
                compilation_options: Default::default(),
                cache: None,
            }),
            mat_mul_nt: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mat_mul_nt"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("mat_mul_nt"),
                compilation_options: Default::default(),
                cache: None,
            }),
            mat_mul_tn: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mat_mul_tn"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("mat_mul_tn"),
                compilation_options: Default::default(),
                cache: None,
            }),
            mat_mul_tt: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mat_mul_tt"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("mat_mul_tt"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group_layout: bind_group_layout,
        }
    }
}

pub struct Operation {
    pub plus_d4: wgpu::ComputePipeline,
    pub minus_d4: wgpu::ComputePipeline,
    pub times_d4: wgpu::ComputePipeline,
    pub div_d4: wgpu::ComputePipeline,

    pub bind_group_layout: BindGroupLayout,
}

impl Operation {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Operation::new"),
            source: wgpu::ShaderSource::Wgsl(include_str!("operation.wgsl").into()),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("Operation::new"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Operation {
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
