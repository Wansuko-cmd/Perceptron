use crate::kernels::task::ComputeTask;

pub struct Dispatcher {
    pub count: usize,
    active_encoder: Option<wgpu::CommandEncoder>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Dispatcher { count: 0, active_encoder: None }
    }

    pub fn dispatch(&mut self, task: ComputeTask, device: &wgpu::Device) {
        let encoder = self.active_encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Dispatcher::dispatch") })
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(task.label),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(task.pipeline);
            compute_pass.set_bind_group(0, &task.bind_group, &[]);

            compute_pass.dispatch_workgroups(task.workgroups[0], task.workgroups[1], task.workgroups[2]);
        }

        self.count += 1;
    }

    pub fn submit(&mut self, queue: &wgpu::Queue) {
        if let Some(encoder) = self.active_encoder.take() {
            queue.submit(Some(encoder.finish()));
            self.active_encoder = None;
            self.count = 0;
        };
    }
}
