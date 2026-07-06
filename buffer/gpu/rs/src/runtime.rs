pub mod dispatcher;
pub mod profiler;

use std::sync::{Arc, Mutex};

use crate::{kernels::{Kernels, task::Task}, runtime::{dispatcher::Dispatcher, profiler::CpuProfiler}};

pub struct Runtime {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub kernels: Kernels,
    pub cpu_profiler: CpuProfiler,
    dispatcher: Mutex<Dispatcher>,
}

impl Runtime {
    pub async fn new(enable_profiler: bool) -> Self {
        let instance = wgpu::Instance::default();
        let request = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;
        let adapter = match request {
            Ok(adapter) => adapter,
            Err(_) => {
                eprintln!("Warning: SW GPU will be used.");
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: true,
                    })
                    .await
                    .unwrap()
            },
        };

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Runtime:new"),
                required_features: wgpu::Features::IMMEDIATES,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    max_buffer_size: limits.max_buffer_size,
                    max_immediate_size: limits.max_immediate_size,
                    ..wgpu::Limits::defaults()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let kernels = Kernels::new(&device);
        let cpu_profiler = CpuProfiler::new(enable_profiler);
        let dispatcher = Mutex::new(Dispatcher::new());

        Runtime {
            device: device,
            queue: queue,
            kernels: kernels,
            cpu_profiler: cpu_profiler,
            dispatcher: dispatcher,
        }
    }
}

impl Runtime {
    const THREHOLD: usize = 100;

    pub fn dispatch<T: Task>(&self, task: T) {
        let mut dispatcher = self.dispatcher.lock().unwrap();

        dispatcher.dispatch(task, &self.device);
        if dispatcher.count >= Runtime::THREHOLD {
            dispatcher.submit(&self.queue);
        }
    }

    pub fn flush(&self) {
        self.dispatcher.lock().unwrap().submit(&self.queue);
    }

    pub fn wait(&self) {
         let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}
