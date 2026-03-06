use wgpu::{Device, Queue, util::DeviceExt};
use bytemuck;
use std::sync::mpsc;

pub struct GPUBuffer {
    pub buffer: wgpu::Buffer,
}

impl GPUBuffer {
    pub const SIZE_BYTES: usize = std::mem::size_of::<f32>();

    pub fn create(size: usize, device: &Device) -> GPUBuffer {
        let byte_size = (size * GPUBuffer::SIZE_BYTES) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPUBuffer::create_buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GPUBuffer { buffer }
    }

    pub fn init(value: &[f32], device: &Device) -> GPUBuffer {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPUBuffer::init_buffer"),
            contents: bytemuck::cast_slice(&value),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        GPUBuffer { buffer }
    }
}

impl GPUBuffer {
    pub fn read_all(self: &Self, dest: &mut [f32], device: &Device, queue: &Queue) {
        let size = self.buffer.size();
        // GPU上の共有バッファを用意
        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPUBufer::read_all"),
            size: size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 対象バッファの値を共有バッファにコピー
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("GPUBuffer:read_all")
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &read_buffer, 0, size);
        queue.submit(Some(encoder.finish()));

        {
            let read_buffer_slice = read_buffer.slice(..);

            // コピー待機処理
            let (sender, receiver) = mpsc::sync_channel(1);
            read_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            receiver.recv().unwrap().unwrap();

            let data = read_buffer_slice.get_mapped_range();
            dest.copy_from_slice( bytemuck::cast_slice(&data));
        }

        read_buffer.unmap();
    }

    pub fn slice(&self, start: usize, end: usize, device: &Device, queue: &Queue) -> GPUBuffer {
        let size = ((end - start) * GPUBuffer::SIZE_BYTES) as u64;
        let offset = (start * GPUBuffer::SIZE_BYTES) as u64;

        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPUBuffer::slice"),
            size: size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPUBuffer::slice"),
        });

        encoder.copy_buffer_to_buffer(&self.buffer, offset, &new_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        GPUBuffer { buffer: new_buffer }
    }

    pub fn copy_into(&self, dest: &GPUBuffer, dest_offset: usize, device: &Device, queue: &Queue) {
        let size = self.buffer.size();
        let offset = (dest_offset * GPUBuffer::SIZE_BYTES) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPUBuffer::copy_into"),
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dest.buffer, offset, size);
        queue.submit(std::iter::once(encoder.finish()));
    }
}

impl GPUBuffer {
    pub fn write(self: &Self, index: usize, value: f32, queue: &Queue) {
        let offset = (index * GPUBuffer::SIZE_BYTES) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(&value));
    }
}

impl GPUBuffer {
    pub fn count(self: &Self) -> usize {
        let byte_size = self.buffer.size() as usize;
        byte_size / GPUBuffer::SIZE_BYTES
    }

    pub fn workgroup_count(self: &Self, workgroup_size: u32) -> [u32; 3] {
        let count = self.count() as u32;
        [(count + workgroup_size - 1) / workgroup_size, 1u32, 1u32]
    }

    pub fn content_equals(&self, other: &GPUBuffer, device: &Device, queue: &Queue) -> bool {
        if self.buffer == other.buffer { return true; }
        if self.buffer.size() != other.buffer.size() { return false; }

        let mut self_data = vec![0.0f32; self.count()];
        let mut other_data = vec![0.0f32; other.count()];
        
        self.read_all(&mut self_data, device, queue);
        other.read_all(&mut other_data, device, queue);

        return self_data == other_data;
    }
}
