use std::sync::Arc;
use crate::core::pipeline::Pipeline;

pub struct Context {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline: Pipeline,
}

impl Context {
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
                label: Some("Context::new"),
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
        let pipeline = Pipeline::new(&device);

        Self {
            device: device,
            queue: queue,
            pipeline: pipeline,
        }
    }
}
