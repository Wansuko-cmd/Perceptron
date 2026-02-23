use crate::core::context::Context;

use wgpu::util::DeviceExt;
use bytemuck;
use std::sync::mpsc;

pub struct GPUBuffer {
    pub buffer: wgpu::Buffer,
}

impl GPUBuffer {
    pub fn create(size: usize, context: &Context) -> GPUBuffer {
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPUBuffer::create_buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GPUBuffer { buffer }
    }

    pub fn init(value: &[f32], context: &Context) -> GPUBuffer {
        let buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPUBuffer::init_buffer"),
            contents: bytemuck::cast_slice(&value),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        GPUBuffer { buffer }
    }

    pub fn read_all(self: &Self, dest: &mut [f32], context: &Context) {
        let size = self.buffer.size();
        // GPU上の共有バッファを用意
        let read_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPUBufer::read_all"),
            size: size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 対象バッファの値を共有バッファにコピー
        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("GPUBuffer:read_all")
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &read_buffer, 0, size);
        context.queue.submit(Some(encoder.finish()));

        {
            let read_buffer_slice = read_buffer.slice(..);

            // コピー待機処理
            let (sender, receiver) = mpsc::sync_channel(1);
            read_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            let _ = context.device.poll(wgpu::PollType::wait_indefinitely());
            receiver.recv().unwrap().unwrap();

            let data = read_buffer_slice.get_mapped_range();
            dest.copy_from_slice( bytemuck::cast_slice(&data));
        }

        read_buffer.unmap();
    }

    pub fn write(self: &Self, index: usize, value: f32, context: &Context) {
        let offset = (index * std::mem::size_of::<f32>()) as u64;
        context.queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(&value));
    }

    pub fn count(self: &Self) -> usize {
        let byte_size = self.buffer.size() as usize;
        byte_size / std::mem::size_of::<f32>()
    }
}
