pub mod dispatcher;

use std::sync::Arc;
use crate::{kernels::Kernels, runtime::dispatcher::Dispatcher};

pub struct Runtime {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub kernels: Kernels,
    pub dispatcher: Dispatcher,
}

impl Runtime {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Runtime:new"),
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let kernels = Kernels::new(&device);
        let dispatcher = Dispatcher::new();

        Runtime {
            device: device,
            queue: queue,
            kernels: kernels,
            dispatcher: dispatcher,
        }
    }
}
