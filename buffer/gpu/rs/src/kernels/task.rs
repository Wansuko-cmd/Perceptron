pub struct ComputeTask<'a> {
    pub label: &'static str,
    pub pipeline: &'a wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub workgroups: [u32; 3],
}
