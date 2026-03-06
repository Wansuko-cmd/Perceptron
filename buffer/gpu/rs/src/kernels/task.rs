pub trait Task {
    fn recode(self, encoder: &mut wgpu::CommandEncoder);
}

pub struct ComputeTask<'a> {
    pub label: &'static str,
    pub pipeline: &'a wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub workgroups: [u32; 3],
}

impl Task for ComputeTask<'_> {
    fn recode(self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        compute_pass.dispatch_workgroups(self.workgroups[0], self.workgroups[1], self.workgroups[2]);
    }
}
